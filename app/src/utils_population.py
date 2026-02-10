"""Functions for population data."""
import copy
import io
import math
import os
import shutil
import tempfile
import time
import urllib.request
from ftplib import FTP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from zipfile import ZipFile

import country_converter as coco
import ee
import geemap
import geopandas as gpd
import numpy as np
import rasterio
import requests
import shapely
import streamlit as st
import streamlit_ext as ste
from pyproj import CRS
from rasterio.features import shapes
from rasterio.mask import mask
from rasterstats import zonal_stats
from shapely.geometry import box, shape
from shapely.ops import unary_union
from src.utils_plotting import display_polygons_on_map
from streamlit_folium import folium_static


def mock_st_text(text: str, verbose: bool, text_on_streamlit: bool) -> None:
    """
    Return streamlit text using a given text object.

    If the object is None, return None. This function can be used in streamlit
    or in jupyter notebook.

    Inputs
    -------
    text (str): text to display.
    verbose (bool): if True, print details run.
    text_on_streamlit (str): if True, print text in the Streamlit app
        via st.write(), otherwise print to console.

    Returns
    -------
    None or the updated text_field.
    """
    if not verbose:
        return
    if text_on_streamlit:
        st.write(text)
    else:
        print(text)


def create_tif_folder(tif_parent_folder: str = "app/data") -> str:
    """
    Create a folder where to store TIF files.

    The folder is created in the parent folder defined as a parameter. The
    function scans the existing subfolders in the parent folder, and define
    the new folder name as 'pop_data_<x+1>', where <x> is the highest suffix
    among the ones already existing.

    Inputs
    -------
    tif_parent_folder (str): The path to the parent directory.

    Returns
    -------
    str: The path to the created folder.

    """
    if not os.path.exists(tif_parent_folder):
        os.makedirs(tif_parent_folder)
    current_suffixes = [
        int(f.name.split("_")[-1])
        for f in os.scandir(tif_parent_folder)
        if f.is_dir()
    ]
    if current_suffixes:
        suffix = str(max(current_suffixes) + 1).zfill(2)
    else:
        suffix = "00"
    tif_folder = os.path.join(tif_parent_folder, f"pop_data_{suffix}")
    if not os.path.exists(tif_folder):
        os.makedirs(tif_folder)
    return tif_folder


def delete_tif_folder(tif_folder: str) -> None:
    """
    Delete the folder where TIF files are stored, together with its content.

    Inputs
    -------
    tif_folder (str): The path to the directory where TIF files are stored.

    Returns
    -------
    None.

    """
    if os.path.exists(tif_folder):
        shutil.rmtree(tif_folder)


def size_on_disk(
    file: Union[st.runtime.uploaded_file_manager.UploadedFile, str]
) -> float:
    """
    Calculate the size of a file on disk.

    Inputs
    -------
    file (str or st.runtime.uploaded_file_manager.UploadedFile): filename or
        output of the function streamlit.file_uploader.

    Returns
    -------
    float: The size of the file in megabytes (MB).

    """
    if isinstance(file, str):
        file_size = os.stat(file).st_blocks * 512
    else:
        file_size = len(file.getvalue())
    return file_size / (1024**2)


def _to_2d(x: float, y: float, z: float) -> tuple:
    """
    Convert shapely.Point to 2d.

    Inputs
    -------
    x (float): The x-coordinate of the point.
    y (float): The y-coordinate of the point.
    z (float): The z-coordinate of the point.

    Returns
    -------
    tuple: A tuple containing the x and y coordinates of the point.

    """
    return tuple(filter(None, [x, y]))


def visualize_data(
    gdf: gpd.GeoDataFrame,
    add_map: bool = False,
    progress_bar: bool = True,
) -> None:
    """
    Visualize data with a table and, optionally, a map.

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame containing the data.
    add_map (bool, optional): if True, add a folium Map showing the polygons.
    progress_bar (bool, optional): if True, visualize a progress bar.

    Returns
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame.
    error (str, optional): error string if error was generated, otherwise None.
    """
    if progress_bar:
        st.markdown("### ")
        placeholder = st.empty()
        progress_text_base = "Operation in progress. Please wait. "
        progress_text = "Loading data..."
        my_bar = placeholder.progress(
            0, text=progress_text_base + progress_text
        )

    st.dataframe(gdf.to_wkt())

    if add_map:
        if progress_bar:
            progress_text = "Creating map..."
            my_bar.progress(0.5, text=progress_text_base + progress_text)

        Map = display_polygons_on_map(gdf, add_countryborders=False)
        folium_static(Map)

    if progress_bar:
        my_bar.progress(1.0, text=" ")
        time.sleep(2)
        placeholder.empty()


def split_single_coordinate(
    min_c: float, max_c: float, width_coordinate: float
) -> list:
    """
    Split a coordinate range into a list of coordinates with a given step size.

    Inputs
    -------
    min_c (float): The minimum value of the coordinate range.
    max_c (float): The maximum value of the coordinate range.
    width_coordinate (float): The step size for splitting the coordinate range.

    Returns
    -------
    list: A list of coordinates starting from min_c and incrementing by
        width_coordinate until max_c.

    """
    c = min_c
    c_list = []
    while c < max_c:
        c_list.append(c)
        c += width_coordinate
    c_list.append(max_c)
    return c_list


def split_2d_coordinates(bounds: list, width_coordinate: float) -> tuple:
    """
    Split a 2D coordinate space into separate lists of x and y coordinates.

    The 2D coordinate space is defined by bounds.

    Inputs
    -------
    bounds (list): The bounds of the coordinate space [min_x, min_y, max_x,
        max_y].
    width_coordinate (float): The step size for splitting the coordinate
        ranges.

    Returns
    -------
    tuple: A tuple containing two lists, x_list and y_list, representing the
        split x and y coordinates.

    """
    min_x = bounds[0]
    min_y = bounds[1]
    max_x = bounds[2]
    max_y = bounds[3]
    x_list = split_single_coordinate(min_x, max_x, width_coordinate)
    y_list = split_single_coordinate(min_y, max_y, width_coordinate)
    return x_list, y_list


def split_polygon(
    pol_original: shapely.Polygon, width_coordinate: float
) -> List[shapely.Polygon]:
    """
    Split a polygon into a list of smaller rectangular polygons.

    Inputs
    -------
    pol_original (shapely.Polygon): The original polygon to be split.
    width_coordinate (float): The step size for splitting the polygon.

    Returns
    -------
    list: A list of smaller rectangular polygons resulting from the split.

    """
    bounds = pol_original.bounds
    x_list, y_list = split_2d_coordinates(bounds, width_coordinate)
    return [
        box(
            *np.array(
                [[x_list[i], y_list[j]], [x_list[i + 1], y_list[j + 1]]]
            ).flatten()
        )
        for i in range(len(x_list) - 1)
        for j in range(len(y_list) - 1)
    ]


def harmonise_projection(
    raster_filename: str,
    keep_old_raster: bool = False,
    verbose: bool = False,
    text_on_streamlit: bool = True,
) -> None:
    """
    Harmonise the projection of a raster file by flipping the y-axis.

    Typically, rasters are defined using a geotransform that starts at the
    upper left coordinate ("north up"), where each subsequent row of the
    raster is moving further south (negative Y). In some cases, rasters
    downloaded from Google Earth Engine are defined using a geotransform that
    starts at the bottom left. This results in the window using to read each
    feature out of the raster being invalid. To be able to be read by libraries
    such as rasterstats, rasters need to be transformed to the tipical format.
    This is achieved by updating the transform property of the raster to be
    anchored from the upper left coordinate, with a negative y cell size.

    Inputs
    -------
    raster_filename (str): The filename of the raster file to be harmonised.
    keep_old_raster (bool, optional): If True, keeps a copy of the original
        raster file. Defaults to False.
    verbose (bool, optional): if True, print details run. Default to False.
    text_on_streamlit (bool, optional): if True, print to streamlit, otherwise
        via the print statement. Only used when verbose is True. Default to
        True.

    Returns
    -------
    None
    """
    with rasterio.open(raster_filename) as old_tif:
        transform = old_tif.profile["transform"]
        if transform.e > 0:
            if keep_old_raster:
                shutil.copyfile(
                    raster_filename, f"{Path(raster_filename).stem}_old.tif"
                )
            new_profile = old_tif.profile.copy()
            new_profile["transform"] = rasterio.Affine(
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                -1 * transform.e,
                transform.f + (transform.e * (old_tif.height - 1)),
            )
            with rasterio.open(raster_filename, "w", **new_profile) as new_tif:
                new_tif.write(np.flipud(old_tif.read(1)), indexes=1)
                mock_st_text(
                    verbose=verbose,
                    text_on_streamlit=text_on_streamlit,
                    text=f"{raster_filename} reprojected",
                )


def load_gdf(
    gdf_file: Union[st.runtime.uploaded_file_manager.UploadedFile, str],
) -> Tuple[gpd.GeoDataFrame, float, Optional[str]]:
    """
    Load GeoDataFrame, change crs, check validity, and make sure it is 2-d.

    Inputs
    -------
    gdf_file (str or st.runtime.uploaded_file_manager.UploadedFile): filename
        of the geopandas.GeoDataFrame.

    Returns
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame.
    file_size (float): size of file on disk
    error (str, optional): error string if error was generated, otherwise None.
    """
    gdf = gpd.read_file(gdf_file)
    gdf.to_crs(CRS.from_user_input(4326), inplace=True)
    if check_gdf_geometry(gdf):
        error = None
    else:
        error = (
            "Error with the shapefile. Either there were problems with the "
            "download, or the dataframe contains geometries that are not "
            "polygons. Check the source data."
        )
    return convert_gdf_to_2d(gdf), size_on_disk(gdf_file), error


def check_gdf_geometry(gdf: gpd.GeoDataFrame) -> bool:
    """Check whether all geometries are polygons or multipolygons.

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame with polygons.

    Returns
    -------
    (bool): True if all geometries are points, False otherwise.
    """
    return all(["Polygon" in geom for geom in gdf.geom_type.tolist()])


def convert_gdf_to_2d(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Make sure that all geometries in GeoDataFrame are 2-dimensional.

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame with polygons.

    Returns
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame with 2-d polygons.
    """
    geom_2d_list = []
    for geom in gdf["geometry"].tolist():
        if "Multi" in geom.geom_type:
            coord_point = list(geom.geoms)[0].exterior.coords[0]
        else:
            coord_point = list(geom.exterior.coords)[0]
        if len(coord_point) > 2:
            geom_2d_list.append(shapely.ops.transform(_to_2d, geom))
        else:
            geom_2d_list.append(geom)

    return gdf.set_geometry(geom_2d_list)


def aggregate_raster_on_geometries(
    raster_filename: str,
    geometry_list: List[shapely.Geometry],
    library: str = "rasterio",
    rounding_to_which_power_of_ten: int = 2,
) -> List[int]:
    """
    Compute zonal statistics of a raster file over a list of vector geometries.

    Inputs:
    -------
    raster_filename (str): filepath or url to the input raster file.
    geometry_list (list): list of shapely geometries (e.g. polygons) over which
        to compute the zonal statistics.
    library (str, optional): library used to perform the zonal statistics.
        Options are 'rasterio', with the method 'mask.mask', and 'rasterstats',
        with the method 'zonal_stats'. Default to 'rasterio'.
    rounding_to_which_power_of_ten (int, optional): power of ten to which the
        population figures are rounded. Default to 2 (hundred).

    Returns:
    -------
    stats_list (list): list of zonal statistics computed for each input
        geometry, in the same order as the input list. Each item in the list is
        an integer (rounded to the next hundred) representing the zonal
        statistics (sum) for a single geometry.
    """
    if library == "rasterstats":
        stats_dict = zonal_stats(geometry_list, raster_filename, stats="sum")
        stats_list = [s["sum"] for s in stats_dict]
    elif library == "rasterio":
        stats_list = []
        with rasterio.open(raster_filename) as raster_file:
            for geom in geometry_list:
                try:
                    masked, _ = mask(
                        dataset=raster_file, shapes=[geom], crop=True
                    )
                    stats_list.append(np.nansum(masked))
                except Exception as e:
                    if "Input shapes do not overlap raster" in str(e):
                        stats_list.append(None)
                    else:
                        raise
    else:
        raise ValueError(f"Library {library} not known.")
    cleaned_stats = []
    for s in stats:
        # Check if value is None, Infinity, or NaN (Not a Number)
        if s is None or math.isinf(s) or math.isnan(s):
            cleaned_stats.append(0) # Treat invalid data as 0 population
        else:
            cleaned_stats.append(int(round(s, -rounding_to_which_power_of_ten)))
            
    return cleaned_stats
    # -------------------------------------


def find_intersections_polygon(
    pol: shapely.Polygon,
    world_gdf: gpd.GeoDataFrame,
) -> Dict:
    """
    Find the intersections between a Polygon object and country borders.

    Inputs
    -------
    pol (shapely.Polygon): shapely Polygon object.
    world_gdf (geopandas.GeoDataFrame): Geodataframe containing polygons of
        world countries.

    Returns
    -------
    intersect_dict (dict): dictionary with
        - keys containing ISO3 country codes corresponding to the countries
        intersected by the input Polygon,
        - values containing intersected portions of each country's polygon, as
        Polygon objects.
    """
    ind_intersect = pol.intersects(world_gdf["geometry"])
    world_intersect = world_gdf[
        world_gdf.index.isin(ind_intersect[ind_intersect].index.values)
    ]

    return dict(
        [
            (
                world_intersect["iso3"].iloc[i],
                world_intersect["geometry"].iloc[i].intersection(pol),
            )
            for i in range(len(world_intersect))
        ]
    )


def find_intersections_gdf(
    gdf: gpd.GeoDataFrame,
    data_folder: str = "app/data",
    tif_folder: str = "app/data/pop_data",
) -> Tuple[Dict, List[str]]:
    """
    Find the intersections between polygons in a dataframe and country borders.

    Country borders are retrieved from WorldPop (1km resolution).

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): GeoDataFrame containing Polygon geometries.
    data_folder (str, optional): folder where auxiliary data was saved (such
        as country borders).
    tif_folder (str, optional): folder where the world borders raster data is
        saved.

    Returns
    -------
    intersect_dict (dict): dictionary where the keys are the indices of the
        input GeoDataFrame and the values are dictionaries of intersected
        country polygons, keyed by ISO3 country codes.
    all_iso3_list (list): list of all ISO3 country codes intersected by any of
        the input polygons.
    """
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    filename = f"{data_folder}/world_borders_wp.geojson"

    if os.path.isfile(filename):
        world_gdf = gpd.read_file(filename)
    else:
        world_gdf = retrieve_country_borders_wp(tif_folder)

    intersect_list = [
        find_intersections_polygon(geom, world_gdf=world_gdf)
        for geom in gdf["geometry"].tolist()
    ]
    intersect_dict = dict(zip(range(len(intersect_list)), intersect_list))
    all_iso3_list = list(
        set().union(*[list(inter.keys()) for inter in intersect_list])
    )  # type: list

    return intersect_dict, all_iso3_list


def retrieve_country_borders_wp(
    tif_folder: str = "app/data/pop_data",
) -> gpd.GeoDataFrame:
    """
    Retrieve country borders from WorldPop.

    Country borders are retrieved from WorldPop (1km resolution) in raster
    format. They are then reformatted into vectors and attached to a
    geopandas.GeoDataFrame. The resulting dataframe has iso3 and geometries as
    columns.

    Inputs
    -------
    tif_folder (str, optional): folder where the world borders raster data is
        saved.

    Returns
    -------
    world_gdf (geopandas.GeoDataFrame): GeoDataFrame containing country borders
        of the world, according to WorldPop.
    """
    tif_url = (
        "https://data.worldpop.org/GIS/Mastergrid/Global_2000_2020/0_Mosaicked"
        "/global_mosaic_1km/global_level0_1km_2000_2020.tif"
    )

    raster_file = f'{tif_folder}/{tif_url.split("/")[-1]}'
    urllib.request.urlretrieve(tif_url, raster_file)

    with rasterio.open(raster_file) as src:
        data = src.read(1, masked=True)

        # Use a generator instead of a list
        shape_gen = (
            (shape(s), v) for s, v in shapes(data, transform=src.transform)
        )
        gdf = gpd.GeoDataFrame(
            dict(zip(["geometry", "value"], zip(*shape_gen))), crs=src.crs
        )

        del data

    cc = coco.CountryConverter()

    unique_iso_num_list = list(
        set([int(iso) for iso in gdf["value"].tolist() if iso != 65535])
    )
    unique_iso_str_list = cc.convert(names=unique_iso_num_list, to="ISO3")

    iso3_list = []
    geometry_list = []
    for i, (iso_num, iso_str) in enumerate(
        zip(unique_iso_num_list, unique_iso_str_list)
    ):
        pol = unary_union(gdf[gdf["value"] == iso_num]["geometry"].tolist())
        iso3_list.append(iso_str)
        geometry_list.append(pol)

    world_gdf = gpd.GeoDataFrame(
        data={"iso3": iso3_list}, geometry=geometry_list, crs=gdf.crs
    )

    world_gdf.iat[
        world_gdf[world_gdf["iso3"] == "not found"].iloc[[0]].index[0], 0
    ] = "KOS"
    return (
        world_gdf[world_gdf["iso3"] != "not found"]
        .sort_values(by="iso3")
        .reset_index(drop=True)
    )


def download_worldpop_iso_tif(
    iso3: str,
    tif_folder: str,
    method: str = "http",
    year: int = 2020,
    data_type: str = "UNadj_constrained",
    aggregated: bool = True,
    clobber: bool = False,
    progress_bar: bool = False,
    step_progression: float = 0.0,
    step_size: float = 0.0,
    my_bar: Optional[Any] = None,
    progress_text: Optional[str] = None,
) -> Tuple[List[str], List[str], float]:
    """
    Download WorldPop raster data for given parameters.

    Inputs:
    -------
    iso3 (str): ISO3 country code.
    tif_folder (str): folder name, where download raster data are saved.
    method (str, optional): download method (http API or ftp service).
        ['http' | 'ftp']. Default to 'http'.
    year (int, optional): year of population data. Default to 2020.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    clobber (bool, optional): if True, overwrite data, if False do not
        download data if already present in the folder. Default to False.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.
    step_progression (float, optional): number that indicates the progression
        of the computation. Values between 0 and 1. Default to 0.
    step_size (float, optional): number that indicates the step size for the
        progress bar. Values between 0 and 1. Default to 0.
    mybar (st.progress, optional): progress bar object. Default to None.
    progress_text (str, optional): text to visualize above the progress bar.
        Default to None.

    Returns:
    -------
    filename_list (list): list of filenames of the GeoTIFF files containing the
        population data for the given parameters.
    label_list (list): list of labels for dataframe columns. If the population
        data is disaggregated, each label indicates gender and age.
    step_progression (float): number that indicates the update progression
        of the computation. Values between 0 and 1.
    """
    allowed_method = ["http", "ftp"]
    if method not in allowed_method:
        raise ValueError(
            "method should be one of the options: " + ", ".join(allowed_method)
        )

    filename_base = f"{tif_folder}/{iso3.lower()}_{year}_{data_type}"

    if method == "http":
        return download_worldpop_iso3_tif_http(
            iso3,
            year=year,
            data_type=data_type,
            aggregated=aggregated,
            filename_base=filename_base,
            clobber=clobber,
            progress_bar=progress_bar,
            step_progression=step_progression,
            step_size=step_size,
            my_bar=my_bar,
            progress_text=progress_text,
        )
    else:
        return download_worldpop_iso3_tif_ftp(
            iso3,
            year=year,
            data_type=data_type,
            aggregated=aggregated,
            filename_base=filename_base,
            clobber=clobber,
            progress_bar=progress_bar,
            step_progression=step_progression,
            step_size=step_size,
            my_bar=my_bar,
            progress_text=progress_text,
        )


def download_worldpop_iso3_tif_http(
    iso3: str,
    data_type: str = "UNadj_constrained",
    aggregated: bool = True,
    filename_base: Optional[str] = None,
    year: int = 2020,
    clobber: bool = False,
    progress_bar: bool = False,
    step_progression: float = 0.0,
    step_size: float = 0.0,
    my_bar: Any = st.progress(0),
    progress_text: Optional[str] = None,
) -> Tuple[List[str], List[str], float]:
    """
    Download WorldPop raster data for given parameters, using the API via http.

    Inputs:
    -------
    iso3 (str): ISO3 country code.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    filename_base (str, optional): part of the filename of the downloaded
        raster file.
    year (int, optional): year of population data. Default to 2020.
    clobber (bool, optional): if True, overwrite data, if False do not
        download data if already present in the folder. Default to False.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.
    step_progression (float, optional): number that indicates the progression
        of the computation. Values between 0 and 1. Default to 0.
    step_size (float, optional): number that indicates the step size for the
        progress bar. Values between 0 and 1. Default to 0.
    mybar (st.progress, optional): progress bar object. Default to None.
    progress_text (str, optional): text to visualize above the progress bar.
        Default to None.

    Returns:
    -------
    filename_list (list): list of filenames of the GeoTIFF files containing the
        population data for the given parameters.
    label_list (list): list of labels for dataframe columns. If the population
        data is disaggregated, each label indicates gender and age.
    step_progression (float): number that indicates the update progression
        of the computation. Values between 0 and 1.
    """
    if aggregated:
        api_suffix_1 = "age_structures/"
    else:
        api_suffix_1 = "pop/"

    if data_type == "unconstrained" and aggregated:
        api_suffix_2 = "wpgp"
    elif data_type == "unconstrained" and not aggregated:
        api_suffix_2 = "aswpgp"
    else:
        if year != 2020:
            raise ValueError("Constrained data is only available for 2020.")
        else:
            if data_type == "constrained" and aggregated:
                api_suffix_2 = f"cic{year}_100m"
            elif data_type == "constrained" and not aggregated:
                api_suffix_2 = f"ascic_{year}"
            elif data_type == "UNadj_constrained" and aggregated:
                api_suffix_2 = f"cic{year}_UNadj_100m"
            elif data_type == "UNadj_constrained" and not aggregated:
                api_suffix_2 = f"ascicua_{year}"

    api_url = (
        f"https://hub.worldpop.org/rest/data/{api_suffix_1}{api_suffix_2}"
        f"?iso3={iso3}"
    )

    response = requests.get(api_url)
    data_json = response.json()["data"]

    if data_type == "unconstrained":
        year_index = [
            i
            for (i, data_year) in enumerate(data_json)
            if int(data_year["popyear"]) == year
        ][0]
    else:
        year_index = 0

    tif_url_list = sorted(data_json[year_index]["files"])

    filename_list = []
    label_list = []

    for j, tif_url in enumerate(tif_url_list):
        if filename_base is None:
            filename = tif_url.split("/")[-1]
            label = f"pop_{j}"
        else:
            if aggregated:
                filename = f"{filename_base}.tif"
                label = "pop_tot"
            else:
                gender = tif_url.split("/")[-1].split("_")[1]
                age = tif_url.split("/")[-1].split("_")[2].zfill(2)
                filename = f"{filename_base}_{gender}_{age}.tif"
                label = f"pop_{gender}_{age}"

        if clobber:
            condition_to_download = True
        else:
            condition_to_download = not os.path.isfile(filename)

        if condition_to_download:
            urllib.request.urlretrieve(tif_url, filename)

        filename_list.append(filename)
        label_list.append(label)

        if progress_bar:
            step_progression += step_size
            my_bar.progress(round(step_progression, 1), text=progress_text)

    filename_list = [
        filename for _, filename in sorted(zip(label_list, filename_list))
    ]
    label_list.sort()

    return filename_list, label_list, step_progression


def download_worldpop_iso3_tif_ftp(
    iso3: str,
    data_type: str = "UNadj_constrained",
    aggregated: bool = True,
    filename_base: Optional[str] = None,
    year: int = 2020,
    clobber: bool = False,
    progress_bar: bool = False,
    step_progression: float = 0.0,
    step_size: float = 0.0,
    my_bar: Any = st.progress(0),
    progress_text: Optional[str] = None,
) -> Tuple[List[str], List[str], float]:
    """
    Download WorldPop raster data for given parameters, using the ftp service.

    Inputs:
    -------
    iso3 (str): ISO3 country code.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    filename_base (str, optional): part of the filename of the downloaded
        raster file.
    year (int, optional): year of population data. Default to 2020.
    clobber (bool, optional): if True, overwrite data, if False do not
        download data if already present in the folder. Default to False.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.
    step_progression (float, optional): number that indicates the progression
        of the computation. Values between 0 and 1. Default to 0.
    step_size (float, optional): number that indicates the step size for the
        progress bar. Values between 0 and 1. Default to 0.
    mybar (st.progress, optional): progress bar object. Default to None.
    progress_text (str, optional): text to visualize above the progress bar.
        Default to None.

    Returns:
    -------
    filename_list (list): list of filenames of the GeoTIFF files containing the
        population data for the given parameters.
    label_list (list): list of labels for dataframe columns. If the population
        data is disaggregated, each label indicates gender and age.
    step_progression (float): number that indicates the update progression
        of the computation. Values between 0 and 1.
    """
    ftp = FTP("ftp.worldpop.org.uk")
    ftp.login()

    if aggregated:
        folder_base = "/GIS/Population/Global_2000_2020"
    else:
        folder_base = "/GIS/AgeSex_structures/Global_2000_2020"

    if data_type == "unconstrained":
        file_suffix = ""
        folder = f"{folder_base}/{year}/{iso3}/"
        ftp.cwd(folder)
        file_list = ftp.nlst()
        if aggregated:
            file_list = [file_list[0]]
    else:
        if year != 2020:
            raise ValueError("Constrained data is only available for 2020.")
        if aggregated:
            file_suffix = f"_{data_type}"
            folder_base = f"{folder_base}_Constrained/{year}/"

            ftp_sources = ["maxar_v1/", "BSGM/"]
            for source in ftp_sources:
                ftp.cwd(f"{folder_base}/{source}")
                if iso3 in ftp.nlst():
                    break
            folder = f"{folder_base}/{source}{iso3}/"
            ftp.cwd(folder)
            file_list = [f"{iso3.lower()}_ppp_{year}{file_suffix}.tif"]
        else:
            if data_type == "constrained":
                folder_base = f"{folder_base}_Constrained"
            else:
                folder_base = f"{folder_base}_Constrained_UNadj"
            folder = f"{folder_base}/{year}/{iso3}/"
            ftp.cwd(folder)
            file_list = ftp.nlst()

    filename_list = []
    label_list = []

    for j, file in enumerate(file_list):
        if filename_base is None:
            filename = file
            label = f"pop_{j}"
        else:
            if aggregated:
                filename = f"{filename_base}.tif"
                label = "pop_tot"
            else:
                gender = file.split("_")[1]
                age = file.split("_")[2].zfill(2)
                filename = f"{filename_base}_{gender}_{age}.tif"
                label = f"pop_{gender}_{age}"

        if clobber:
            condition_to_download = True
        else:
            condition_to_download = not os.path.isfile(filename)

        if condition_to_download:
            with open(filename, "wb") as file_out:
                ftp.retrbinary(f"RETR {file}", file_out.write)

        filename_list.append(filename)
        label_list.append(label)

        if progress_bar:
            step_progression += step_size
            my_bar.progress(round(step_progression, 1), text=progress_text)

    filename_list = [
        filename for _, filename in sorted(zip(label_list, filename_list))
    ]
    label_list.sort()

    ftp.quit()

    return filename_list, label_list, step_progression


def add_population_data_from_wpAPI(
    gdf: gpd.GeoDataFrame,
    data_type: str = "UNadj_constrained",
    year: int = 2020,
    aggregated: bool = True,
    data_folder: str = "app/data",
    tif_folder: str = "app/data/pop_data",
    clobber: bool = False,
    rounding_to_which_power_of_ten: int = 2,
    verbose: bool = False,
    text_on_streamlit: bool = True,
    progress_bar: bool = False,
) -> gpd.GeoDataFrame:
    """
    Add population data to a GeoDataFrame by aggregating WorldPop data.

    Since WorldPop population data can only be downloaded per country using the
    WorldPop API, the function calculates the intersections between the
    geometries in the dataframe and each one of the world countries. Then it
    calculates zonal statistics for each pair (geometry, country) and aggregate
    the results per geometry. This function is not used in the Streamlit app,
    as it requires a large amount of computational time and memory.

    Inputs
    ----------
    gdf (geopandas.GeoDataFram: GeoDataFrame containing the geometries for
        which to add population data.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    year (int, optional): year of population data. Default to 2020.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    data_folder (str, optional): folder where auxiliary data was saved (such
        as country borders).
    tif_folder (str, optional): folder where the population raster data is
        saved.
    clobber (bool, optional): if True, overwrite data, if False do not
        download data if already present in the folder. Default to False.
    rounding_to_which_power_of_ten (int, optional): power of ten to which the
        population figures are rounded. Default to 2 (hundred).
    verbose (bool, optional): if True, print details run. Default to False.
    text_on_streamlit (bool, optional): if True, print to streamlit, otherwise
        via the print statement. Only used when verbose is True.
        Default to True.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.
        Default to False.

    Returns
    -------
    gdf_with_pop (geopandas.GeoDataFrame): input GeoDataFrame with additional
        columns containing the (dis)aggregated population data.
    """
    progress_text_base = "Operation in progress. Please wait. "

    progress_text = "Finding country intersections..."
    my_bar = st.progress(0, text=progress_text_base + progress_text)

    gdf_with_pop = gdf.copy()

    allowed_data_type = ["unconstrained", "constrained", "UNadj_constrained"]
    if data_type not in allowed_data_type:
        raise ValueError(
            "data_type should be one of the options: "
            + ", ".join(allowed_data_type)
        )

    mock_st_text(
        verbose=verbose,
        text_on_streamlit=text_on_streamlit,
        text="Finding intersections with countries...",
    )
    pop_total_dict = {}  # type: dict
    intersect_dict, all_iso3_list = find_intersections_gdf(
        gdf, data_folder=data_folder, tif_folder=tif_folder
    )
    intersect_countries_str = ", ".join(all_iso3_list)
    mock_st_text(
        verbose=verbose,
        text_on_streamlit=text_on_streamlit,
        text=f"Intersected countries: {intersect_countries_str}.",
    )

    if not os.path.exists(tif_folder):
        os.makedirs(tif_folder)

    if progress_bar:
        if aggregated:
            number_of_steps = len(all_iso3_list)
        else:
            number_of_steps = len(all_iso3_list) * 36

        step_progression = 0.1
        step_size = 0.9 / number_of_steps
        progress_text_raster = "Working with country rasters... "
        my_bar.progress(
            round(step_progression, 1),
            text=progress_text_base + progress_text_raster,
        )
    else:
        step_progression = 0.0
        step_size = 0.0

    for i in range(len(all_iso3_list)):
        iso3 = all_iso3_list[i]
        mock_st_text(
            verbose=verbose,
            text_on_streamlit=text_on_streamlit,
            text=f"Country: {iso3}.",
        )
        iso3_list_indexes = [
            i for i, value in intersect_dict.items() if iso3 in value.keys()
        ]
        iso3_list_geometries = [
            value[iso3]
            for i, value in intersect_dict.items()
            if iso3 in value.keys()
        ]

        mock_st_text(
            verbose=verbose,
            text_on_streamlit=text_on_streamlit,
            text="Downloading raster...",
        )

        if progress_bar:
            progress_text = (
                progress_text_base + progress_text_raster + f" {iso3}"
            )

        (
            raster_file_list,
            label_list,
            step_progression,
        ) = download_worldpop_iso_tif(
            iso3,
            tif_folder=tif_folder,
            data_type=data_type,
            year=year,
            aggregated=aggregated,
            clobber=clobber,
            progress_bar=progress_bar,
            step_progression=step_progression,
            step_size=step_size,
            my_bar=my_bar,
            progress_text=progress_text,
        )

        pop_partial_dict = {}

        mock_st_text(
            verbose=verbose,
            text_on_streamlit=text_on_streamlit,
            text="Aggregating raster...",
        )
        for raster_file, label in zip(raster_file_list, label_list):
            if i == 0:
                pop_total_dict[label] = {}

            pop_iso3_agg = aggregate_raster_on_geometries(
                raster_filename=raster_file,
                geometry_list=iso3_list_geometries,
                rounding_to_which_power_of_ten=rounding_to_which_power_of_ten,
            )
            pop_partial_dict[label] = dict(
                zip(iso3_list_indexes, pop_iso3_agg)
            )

            for key, value in pop_partial_dict[label].items():
                if value is not None:
                    if key not in pop_total_dict[label]:
                        pop_total_dict[label][key] = value
                    else:
                        pop_total_dict[label][key] += value

    for label in label_list:
        gdf_with_pop[label] = [
            pop_total_dict[label][i]
            if i in pop_total_dict[label].keys()
            else 0
            for i in gdf.index
        ]

    if progress_bar:
        my_bar.progress(1.0, text=" ")

    mock_st_text(
        verbose=verbose, text_on_streamlit=text_on_streamlit, text="Done!"
    )

    return gdf_with_pop


def add_population_data_from_GEE(
    gdf: gpd.GeoDataFrame,
    size_gdf: float,
    data_type: str = "UNadj_constrained",
    tif_folder: str = "app/data/pop_data",
    year: int = 2020,
    aggregated: bool = True,
    rounding_to_which_power_of_ten: int = 2,
    verbose: bool = False,
    text_on_streamlit: bool = True,
    progress_bar: bool = False,
) -> gpd.GeoDataFrame:
    """
    Add population data to a GeoDataFrame using Google Earth Engine.

    The function retrieves population data by clipping rasters retrieved from
    Google Earth Engine (GEE). The GEE Python client must be initialised before
    running this function. There are two ways to retrieve data from GEE:
    (1) with simple geometries, all processes (including clipping the TIF files
        and calculating zonal statistics) can be run on the server side, and
        the final figures are eventually read into Python variables.
    (2) GEE does not accept queries with large geometries, therefore, in this
        case, images clipped according to a bounding box of all geometries are
        downloaded as TIF files (thanks to the library geemap) and these are
        subsequently clipped according to the original geometries.
    The second method takes much longer than the first one, as a further
    limitation is that the image, before being downloaded needs to be split
    into rectangles. This is necessary to guarantee that not too much memory is
    used in the process.
    Here, we first check the size on disk of the GeoDataFrame. It this is
    smaller than 10MB, then we try to retrieve data using the method (1),
    and, in case the limitations of GEE are hit, method (2) is applied. If the
    size is greater than 10MB, method (2) is directly applied.

    Inputs
    ----------
    gdf (geopandas.GeoDataFram: GeoDataFrame containing the geometries for
        which to add population data.
    size_gdf (float): size on disk of the GeoDataFrame, in MB.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    tif_folder (str, optional): folder where the population raster data is
        saved.
    year (int, optional): year of population data. Default to 2020.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    rounding_to_which_power_of_ten (int, optional): power of ten to which the
        population figures are rounded. Default to 2 (hundred).
    verbose (bool, optional): if True, print details run. Default to False.
    text_on_streamlit (bool, optional): if True, print to streamlit, otherwise
        via the print statement. Only used when verbose is True.
        Default to True.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.
        Default to False.

    Returns
    -------
    gdf_with_pop (geopandas.GeoDataFrame): input GeoDataFrame with additional
        columns containing the (dis)aggregated population data.
    """
    alert_message = (
        "The GeoDataFrame uploaded is constituted by complex geometries, "
        "therefore the process of retrieving data from Google Earth Engine "
        "will be longer than expected."
    )
    if size_gdf < 10:
        try:
            return add_population_data_from_GEE_simple_geometries(
                gdf=gdf,
                data_type=data_type,
                year=year,
                aggregated=aggregated,
                rounding_to_which_power_of_ten=rounding_to_which_power_of_ten,
                verbose=verbose,
                text_on_streamlit=text_on_streamlit,
                progress_bar=progress_bar,
            )
        except ee.EEException:
            st.markdown("### ")
            st.write(alert_message)
            return add_population_data_from_GEE_complex_geometries(
                gdf=gdf,
                data_type=data_type,
                tif_folder=tif_folder,
                year=year,
                aggregated=aggregated,
                rounding_to_which_power_of_ten=rounding_to_which_power_of_ten,
                verbose=verbose,
                text_on_streamlit=text_on_streamlit,
                progress_bar=progress_bar,
            )
    else:
        st.write(alert_message)
        return add_population_data_from_GEE_complex_geometries(
            gdf=gdf,
            data_type=data_type,
            tif_folder=tif_folder,
            year=year,
            aggregated=aggregated,
            rounding_to_which_power_of_ten=rounding_to_which_power_of_ten,
            verbose=verbose,
            text_on_streamlit=text_on_streamlit,
            progress_bar=progress_bar,
        )


def add_population_data_from_GEE_simple_geometries(
    gdf: gpd.GeoDataFrame,
    data_type: str = "UNadj_constrained",
    year: int = 2020,
    aggregated: bool = True,
    rounding_to_which_power_of_ten: int = 2,
    verbose: bool = False,
    text_on_streamlit: bool = True,
    progress_bar: bool = False,
) -> gpd.GeoDataFrame:
    """
    Add population data to a GeoDataFrame using GEE for simple geometries.

    The function retrieves population data by clipping rasters retrieved from
    Google Earth Engine. The GEE Python client must be initialised before
    running this function. All processes (including clipping the TIF files
    and calculating zonal statistics) are run on the server side, and the final
    figures are eventually read into Python variables.

    Inputs
    ----------
    gdf (geopandas.GeoDataFram: GeoDataFrame containing the geometries for
        which to add population data.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    year (int, optional): year of population data. Default to 2020.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    rounding_to_which_power_of_ten (int, optional): power of ten to which the
        population figures are rounded. Default to 2 (hundred).
    verbose (bool, optional): if True, print details run. Default to False.
    text_on_streamlit (bool, optional): if True, print to streamlit, otherwise
        via the print statement. Only used when verbose is True.
        Default to True.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.
        Default to False.

    Returns
    -------
    gdf_with_pop (geopandas.GeoDataFrame): input GeoDataFrame with additional
        columns containing the (dis)aggregated population data.
    """
    progress_text_base = "Operation in progress. Please wait. "
    progress_text = "Retrieving data from Google Earth Engine..."

    if progress_bar:
        my_bar = st.progress(0.1, text=progress_text_base + progress_text)

    mock_st_text(
        verbose=verbose,
        text_on_streamlit=text_on_streamlit,
        text=progress_text,
    )

    if data_type == "unconstrained" and aggregated:
        pop_collection = ee.ImageCollection("WorldPop/GP/100m/pop")
        initial_band = 0
    elif data_type == "unconstrained" and not aggregated:
        pop_collection = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")
        initial_band = 1
    elif data_type == "UNadj_constrained" and not aggregated:
        pop_collection = ee.ImageCollection(
            "WorldPop/GP/100m/pop_age_sex_cons_unadj"
        )
        initial_band = 1
    else:
        raise ValueError(
            f"The combination data_type={data_type} and aggregated="
            f"{aggregated} does not exist on Google Earth Engine."
        )

    pop_year = pop_collection.filterMetadata("year", "equals", year)
    bands = pop_year.first().bandNames().getInfo()[initial_band:]

    polygons_fc = ee.FeatureCollection(gdf.__geo_interface__)

    # Each band corresponds to a gender-age category for disaggregated data,
    # or to the total population for aggregated data
    for i, band in enumerate(bands):
        if progress_bar:
            my_bar.progress(
                0.1 + (i + 1) * 0.8 / len(bands),
                text=progress_text_base
                + progress_text
                + f" band {i+1}/{len(bands)}",
            )

        mock_st_text(
            verbose=verbose,
            text_on_streamlit=text_on_streamlit,
            text=f"Band ({i+1}/{len(bands)}): {band}",
        )

        pop_band = pop_year.select(band)
        pop_band_img = pop_band.mosaic()

        scale = pop_band.first().projection().nominalScale().getInfo()

        zonal_statistics = pop_band_img.reduceRegions(
            reducer=ee.Reducer.sum(), collection=polygons_fc, scale=scale
        )

        if i == 0:
            zonal_tot = copy.deepcopy(zonal_statistics)
        else:
            zonal_tot = zonal_tot.merge(zonal_statistics)

    pop_data = zonal_tot.aggregate_array("sum").getInfo()

    if aggregated:
        label = "pop_tot"
        pop_dict = {}
        pop_dict[label] = pop_data
    else:
        # Divide population data in chuncks with length = len(gdf)
        pop_data_dis = [
            pop_data[i * len(gdf) : (i + 1) * len(gdf)]
            for i in range((len(pop_data) + len(gdf) - 1) // len(gdf))
        ]
        labels = [
            f'pop_{band.split("_")[0].lower()}_{band.split("_")[1].zfill(2)}'
            for band in bands
        ]
        pop_dict = {
            label: pop_data for (label, pop_data) in zip(labels, pop_data_dis)
        }

    pop_dict = dict(sorted(pop_dict.items()))

    gdf_with_pop = gdf.copy()

    for label, pop_data in pop_dict.items():
        gdf_with_pop[label] = [
            int(round(pop, -rounding_to_which_power_of_ten))
            for pop in pop_data
        ]

    if progress_bar:
        my_bar.progress(1.0, text=" ")

    mock_st_text(
        verbose=verbose, text_on_streamlit=text_on_streamlit, text="Done!"
    )

    return gdf_with_pop


def add_population_data_from_GEE_complex_geometries(
    gdf: gpd.GeoDataFrame,
    data_type: str = "UNadj_constrained",
    tif_folder: str = "app/data/pop_data",
    year: int = 2020,
    aggregated: bool = True,
    width_coordinate: float = 20,
    rounding_to_which_power_of_ten: int = 2,
    verbose: bool = False,
    text_on_streamlit: bool = True,
    progress_bar: bool = False,
) -> gpd.GeoDataFrame:
    """
    Add population data to a GeoDataFrame using GEE for complex geometries.

    The function retrieves population data by clipping rasters retrieved from
    Google Earth Engine. The GEE Python client must be initialised before
    running this function. GEE does not accept queries with large geometries,
    therefore, in this case, images clipped according to a bounding box of
    all geometries are downloaded as TIF files (thanks to the library geemap)
    and these are subsequently clipped according to the original geometries.
    Each image, before being downloaded, is split into rectangles, whose size
    is defined by the variable `width_coordinate`. This is necessary to
    guarantee that not too much memory is used in the process.

    Inputs
    ----------
    gdf (geopandas.GeoDataFram: GeoDataFrame containing the geometries for
        which to add population data.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    tif_folder (str, optional): folder where the population raster data is
        saved.
    year (int, optional): year of population data. Default to 2020.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    width_coordinate (float, optional): maximum width (and height) of the
        rectangles (in degrees) into which the image is split before being
        downloaded. Default to 20.
    rounding_to_which_power_of_ten (int, optional): power of ten to which the
        population figures are rounded. Default to 2 (hundred).
    verbose (bool, optional): if True, print details run. Default to False.
    text_on_streamlit (bool, optional): if True, print to streamlit, otherwise
        via the print statement. Only used when verbose is True.
        Default to True.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.
        Default to False.

    Returns
    -------
    gdf_with_pop (geopandas.GeoDataFrame): input GeoDataFrame with additional
        columns containing the (dis)aggregated population data.
    """
    bounds = gdf.total_bounds
    bounding_box = box(*bounds)
    pol_list = split_polygon(bounding_box, width_coordinate)

    if data_type == "unconstrained" and aggregated:
        pop_collection = ee.ImageCollection("WorldPop/GP/100m/pop")
        initial_band = 0
    elif data_type == "unconstrained" and not aggregated:
        pop_collection = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")
        initial_band = 1
    elif data_type == "UNadj_constrained" and not aggregated:
        pop_collection = ee.ImageCollection(
            "WorldPop/GP/100m/pop_age_sex_cons_unadj"
        )
        initial_band = 1
    else:
        raise ValueError(
            f"The combination data_type={data_type} and aggregated="
            f"{aggregated} does not exist on Google Earth Engine."
        )

    # We estimated that the computation for each polygon with size 20x20
    # degree^2 takes 6 minutes. If width_coordinate differs from 20, this
    # number should change.
    expected_time = 6 * len(pol_list) if aggregated else 6 * len(pol_list) * 36

    if expected_time < 60:
        st.write(
            "The expected duration of the computation is a maximum of "
            f"{expected_time} minutes."
        )
    else:
        st.write(
            "The expected duration of the computation is a maximum of "
            f"{math.floor(expected_time/60)} hours, "
            f"{round(expected_time%60, -1)} minutes."
        )

    progress_text_base = "Operation in progress. Please wait. "
    progress_text = "Retrieving data from Google Earth Engine..."

    if progress_bar:
        my_bar = st.progress(0.1, text=progress_text_base + progress_text)

    mock_st_text(
        verbose=verbose,
        text_on_streamlit=text_on_streamlit,
        text=progress_text,
    )

    pop_year = pop_collection.filterMetadata("year", "equals", year)
    bands = pop_year.first().bandNames().getInfo()[initial_band:]

    pol_gdf = gpd.GeoDataFrame(index=[0], crs=gdf.crs, geometry=[bounding_box])
    pol_fc = ee.FeatureCollection(pol_gdf.__geo_interface__)

    pol_fc_list = []
    pol_geom_list = []

    for pol in pol_list:
        pol_gdf = gpd.GeoDataFrame(index=[0], crs=gdf.crs, geometry=[pol])
        pol_fc = ee.FeatureCollection(pol_gdf.__geo_interface__)
        pol_geom = pol_fc.geometry()
        pol_fc_list.append(pol_fc)
        pol_geom_list.append(pol_geom)

    zonal_statistics = {}
    gdf_with_pop = gdf.copy()

    if os.path.exists(tif_folder):
        for file in os.listdir(tif_folder):
            if ".tif" in file:
                os.remove(os.path.join(tif_folder, file))

    # Each band corresponds to a gender-age category for disaggregated data,
    # or to the total population for aggregated data
    for i, band in enumerate(bands):
        mock_st_text(
            verbose=verbose,
            text_on_streamlit=text_on_streamlit,
            text=f"Band ({i+1}/{len(bands)}): {band}",
        )

        pop_band = pop_year.select(band)
        pop_band_img = pop_band.mosaic()

        scale = pop_band.first().projection().nominalScale().getInfo()
        crs = pop_band_img.get("system:bands").getInfo()[band]["crs"]

        zonal_statistics[band] = [0] * len(gdf)

        for j in range(len(pol_list)):
            mock_st_text(
                verbose=verbose,
                text_on_streamlit=text_on_streamlit,
                text=f"Polygon: {j+1}/{len(pol_list)}",
            )

            if progress_bar:
                my_bar.progress(
                    0.1
                    + (j + 1) * (i + 1) * 0.8 / (len(bands) * (len(pol_list))),
                    text=progress_text_base
                    + progress_text
                    + (
                        f" band {i+1}/{len(bands)}, "
                        f"polygon {j+1}/{len(pol_list)}"
                    ),
                )

            clip = pop_band_img.clipToCollection(pol_fc_list[j])

            filename = os.path.join(
                tif_folder,
                (
                    f"pol_width_coordinate_{width_coordinate}_"
                    f"band_{str(i+1).zfill(2)}_"
                    f"polygon_{str(j+1).zfill(2)}.tif"
                ),
            )

            if not os.path.exists(tif_folder):
                os.makedirs(tif_folder)

            geemap.download_ee_image(
                image=clip,
                filename=filename,
                region=pol_geom_list[j],
                scale=scale,
                crs=crs,
            )

            harmonise_projection(filename, keep_old_raster=False)

            stats = aggregate_raster_on_geometries(
                raster_filename=filename,
                geometry_list=gdf.geometry.tolist(),
                rounding_to_which_power_of_ten=rounding_to_which_power_of_ten,
            )

            os.remove(filename)

            zonal_statistics[band] = [
                zonal_statistics[band][k] + stats[k]
                if stats[k] is not None
                else zonal_statistics[band][k]
                for k in range(len(stats))
            ]

            if aggregated:
                label = "pop_tot"
            else:
                label = (
                    f"pop_{band.split('_')[0].lower()}_"
                    f"{band.split('_')[1].zfill(2)}"
                )

            gdf_with_pop[label] = zonal_statistics[band]

    if progress_bar:
        my_bar.progress(1.0, text=" ")

    mock_st_text(
        verbose=verbose, text_on_streamlit=text_on_streamlit, text="Done!"
    )

    return gdf_with_pop


def add_population_data(
    gdf: gpd.GeoDataFrame,
    size_gdf: float,
    data_type: str = "UNadj_constrained",
    year: int = 2020,
    aggregated: bool = True,
    data_folder: str = "app/data",
    tif_folder: str = "app/data/pop_data",
    clobber: bool = False,
    rounding_to_which_power_of_ten: int = 2,
    verbose: bool = False,
    text_on_streamlit: bool = True,
    progress_bar: bool = False,
    method: str = "GEE",
) -> gpd.GeoDataFrame:
    """
    Add population data to a GeoDataFrame.

    The function will use, by default, the Google Earth Engine to retrieve
    population data. The other available method is the WorldPop API.

    Inputs
    ----------
    gdf (geopandas.GeoDataFram: GeoDataFrame containing the geometries for
        which to add population data.
    size_gdf (float): size on disk of the GeoDataFrame, in MB.
    data_type (str, optional): type of population estimate.
        ['unconstrained'| 'constrained' | 'UNadj_constrained']. Default to
        'UNadj_constrained'.
    year (int, optional): year of population data. Default to 2020.
    aggregated (bool, optional): if False, download disaggregated data (by
        gender and age), if True download only total population figure.
        Default to True.
    data_folder (str, optional): folder where auxiliary data was saved (such
        as country borders).
    tif_folder (str, optional): folder where the population raster data is
        saved.
    clobber (bool, optional): if True, overwrite data, if False do not
        download data if already present in the folder. Default to False.
    rounding_to_which_power_of_ten (int, optional): power of ten to which the
        population figures are rounded. Default to 2 (hundred).
    verbose (bool, optional): if True, print details run. Default to False.
    text_on_streamlit (bool, optional): if True, print to streamlit, otherwise
        via the print statement. Only used when verbose is True.
        Default to True.
    progress_bar (bool, optional): if True, create a progress bar in Streamlit.
        Default to False.
    method (str): method used to download data. 'GEE' for Google Earth Engine,
        'wpAPI' for WorldPop API. Default to 'GEE'.

    Returns
    -------
    gdf_with_pop (geopandas.GeoDataFrame): input GeoDataFrame with additional
        columns containing the (dis)aggregated population data.
    """
    if method == "wpAPI":
        return add_population_data_from_wpAPI(
            gdf=gdf,
            data_type=data_type,
            year=year,
            aggregated=aggregated,
            data_folder=data_folder,
            tif_folder=tif_folder,
            clobber=clobber,
            rounding_to_which_power_of_ten=rounding_to_which_power_of_ten,
            verbose=verbose,
            text_on_streamlit=text_on_streamlit,
            progress_bar=progress_bar,
        )
    else:
        return add_population_data_from_GEE(
            gdf=gdf,
            size_gdf=size_gdf,
            data_type=data_type,
            tif_folder=tif_folder,
            year=year,
            aggregated=aggregated,
            rounding_to_which_power_of_ten=rounding_to_which_power_of_ten,
            verbose=verbose,
            text_on_streamlit=text_on_streamlit,
            progress_bar=progress_bar,
        )


def save_shapefile_with_bytesio(
    gdf: gpd.GeoDataFrame,
    directory: str,
) -> str:
    """
    Create zipped shapefile using a GeoDataFrame as an input.

    A zipped shapefile, as well as a shapefile with extension .shp, are saved
    in the folder defined by the argument `directory`.

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): input data.
    directory (str): filepath where the files are saved.

    Returns
    -------
    zip_filename (str): filename of the zip file.
    """
    zip_filename = "user_shapefiles_zip.zip"
    gdf.to_file(f"{directory}/user_shapefiles.shp", driver="ESRI Shapefile")
    zipObj = ZipFile(f"{directory}/{zip_filename}", "w")
    zipObj.write(
        f"{directory}/user_shapefiles.shp", arcname="user_shapefiles.shp"
    )
    zipObj.write(
        f"{directory}/user_shapefiles.cpg", arcname="user_shapefiles.cpg"
    )
    zipObj.write(
        f"{directory}/user_shapefiles.dbf", arcname="user_shapefiles.dbf"
    )
    zipObj.write(
        f"{directory}/user_shapefiles.prj", arcname="user_shapefiles.prj"
    )
    zipObj.write(
        f"{directory}/user_shapefiles.shx", arcname="user_shapefiles.shx"
    )
    zipObj.close()

    return zip_filename


def st_download_shapefile(
    gdf: gpd.GeoDataFrame,
    filename: str,
    label: str = "Download shapefile",
) -> None:
    """
    Create a button to download a shapefile with Streamlit.

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): input data.
    filename (str): name of the saved file.
    label (str, optional): button label. Default to "Download shapefile".
    """
    with tempfile.TemporaryDirectory() as tmp:
        # create the shape files in the temporary directory
        zip_filename = save_shapefile_with_bytesio(gdf, tmp)
        with open(f"{tmp}/{zip_filename}", "rb") as file:
            ste.download_button(
                label=label,
                data=file,
                file_name=filename,
                mime="application/zip",
            )


def st_download_csv(
    gdf: gpd.GeoDataFrame,
    filename: str,
    label: str = "Download csv",
) -> None:
    """
    Create a button to download a shapefile with Streamlit.

    Inputs
    -------
    gdf (geopandas.GeoDataFrame): input data.
    filename (str): name of the saved file.
    label (str, optional): button label. Default to "Download shapefile".
    """
    gdf = gdf.drop(columns="geometry")
    buffer = io.BytesIO()
    gdf.to_csv(buffer, index=False)
    ste.download_button(
        label=label,
        data=buffer,
        file_name=f"{filename}",
        mime="text/csv",
    )
