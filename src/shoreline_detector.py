"""
Shoreline Detection Module

Detects shoreline positions from processed satellite imagery using various
algorithms including water index thresholding, edge detection, and machine learning.
"""

import logging
from typing import Optional, Tuple, List

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure
from skimage.segmentation import active_contour
import cv2
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import LineString, shape
import pandas as pd


class ShorelineDetector:
    """
    Detects shoreline positions from satellite imagery.
    
    Attributes:
        logger (logging.Logger): Logger instance
        config (dict): Configuration parameters
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the shoreline detector.
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or self._default_config()
    
    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'method': 'water_index',
            'ndwi_threshold': 0.3,
            'edge_algorithm': 'canny',
            'min_shoreline_length_m': 100,
            'buffer_distance_m': 50,
            'smoothing_kernel_size': 5
        }
    
    def detect_shoreline(self, water_index: np.ndarray, 
                        method: str = 'threshold') -> np.ndarray:
        """
        Detect shoreline from water index or image data.
        
        Args:
            water_index: Water index array (NDWI, MNDWI, etc.)
            method: Detection method ('threshold', 'edge', 'contour')
            
        Returns:
            Binary shoreline mask
        """
        self.logger.info(f"Detecting shoreline using {method} method")
        
        if method == 'threshold':
            shoreline = self._threshold_detection(water_index)
        elif method == 'edge':
            shoreline = self._edge_detection(water_index)
        elif method == 'contour':
            shoreline = self._contour_detection(water_index)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Apply post-processing
        shoreline = self._post_process(shoreline)
        
        return shoreline
    
    def _threshold_detection(self, water_index: np.ndarray) -> np.ndarray:
        """
        Detect shoreline using threshold method.
        
        Water pixels typically have positive water index values.
        Shoreline is where index crosses threshold.
        
        Args:
            water_index: Water index array
            
        Returns:
            Binary shoreline mask
        """
        threshold = self.config.get('ndwi_threshold', 0.3)
        
        # Create binary water mask
        water_mask = water_index > threshold
        
        # Find shoreline as edge of water mask
        edges = ndimage.sobel(water_mask.astype(float))
        shoreline = edges > 0.1
        
        self.logger.info(f"Threshold detection: {np.sum(shoreline)} pixels detected")
        return shoreline
    
    def _edge_detection(self, water_index: np.ndarray) -> np.ndarray:
        """
        Detect shoreline using edge detection algorithms.
        
        Args:
            water_index: Water index array
            
        Returns:
            Binary shoreline mask
        """
        # Normalize to 0-255 range
        normalized = ((water_index - water_index.min()) / 
                     (water_index.max() - water_index.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(normalized, 50, 150)
        
        self.logger.info(f"Edge detection: {np.sum(edges > 0)} pixels detected")
        return edges > 0
    
    def _contour_detection(self, water_index: np.ndarray) -> np.ndarray:
        """
        Detect shoreline using active contour (snake) method.
        
        Args:
            water_index: Water index array
            
        Returns:
            Binary shoreline mask
        """
        # Smooth the image
        smoothed = filters.gaussian(water_index, sigma=2)
        
        # Find contours using OpenCV
        contours, _ = cv2.findContours(
            ((smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-8) * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Create binary mask
        shoreline = np.zeros_like(water_index, dtype=bool)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                cv2.drawContours(shoreline, [contour], 0, True, 1)
        
        self.logger.info(f"Contour detection: {len(contours)} contours found")
        return shoreline
    
    def _post_process(self, shoreline: np.ndarray) -> np.ndarray:
        """
        Post-process detected shoreline.
        
        Args:
            shoreline: Binary shoreline mask
            
        Returns:
            Post-processed shoreline
        """
        # Remove small noise
        shoreline = morphology.remove_small_objects(shoreline, min_size=50)
        
        # Fill small holes
        shoreline = ndimage.binary_fill_holes(shoreline)
        
        # Apply morphological operations
        shoreline = morphology.binary_closing(shoreline, morphology.disk(3))
        shoreline = morphology.binary_opening(shoreline, morphology.disk(2))
        
        return shoreline
    
    def vectorize_shoreline(self, shoreline_mask: np.ndarray, 
                           geotransform: Tuple = None) -> gpd.GeoDataFrame:
        """
        Convert raster shoreline to vector format (GeoDataFrame).
        
        Args:
            shoreline_mask: Binary shoreline raster
            geotransform: Rasterio transform object
            
        Returns:
            GeoDataFrame with shoreline geometries
        """
        self.logger.info("Vectorizing shoreline to geometries")
        
        # Label connected components
        labeled, num_features = ndimage.label(shoreline_mask)
        
        geometries = []
        properties = []
        
        for label_id in range(1, num_features + 1):
            # Get mask for this component
            component_mask = labeled == label_id
            
            # Find contours
            contours = measure.find_contours(component_mask, 0.5)
            
            for contour in contours:
                if len(contour) > 10:  # Minimum points for valid line
                    # Convert to coordinates (swap rows/cols)
                    coords = [(pt[1], pt[0]) for pt in contour]
                    
                    # Apply geotransform if provided
                    if geotransform is not None:
                        coords = self._apply_geotransform(coords, geotransform)
                    
                    geometry = LineString(coords)
                    
                    if geometry.length > self.config.get('min_shoreline_length_m', 100):
                        geometries.append(geometry)
                        properties.append({'shoreline_id': len(geometries)})
        
        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
        self.logger.info(f"Vectorized {len(geometries)} shoreline features")
        
        return gdf
    
    def _apply_geotransform(self, coords: List[Tuple], geotransform) -> List[Tuple]:
        """
        Apply geotransform to pixel coordinates.
        
        Args:
            coords: List of (x, y) pixel coordinates
            geotransform: Rasterio transform
            
        Returns:
            List of (lon, lat) geographic coordinates
        """
        transformed = []
        for x, y in coords:
            geo_x = geotransform.c + x * geotransform.a + y * geotransform.b
            geo_y = geotransform.f + x * geotransform.d + y * geotransform.e
            transformed.append((geo_x, geo_y))
        return transformed
    
    def smooth_shoreline(self, gdf: gpd.GeoDataFrame, 
                        method: str = 'gaussian') -> gpd.GeoDataFrame:
        """
        Smooth shoreline geometries.
        
        Args:
            gdf: GeoDataFrame with shoreline geometries
            method: Smoothing method ('gaussian', 'chaikin', 'rdp')
            
        Returns:
            GeoDataFrame with smoothed geometries
        """
        self.logger.info(f"Smoothing shoreline using {method} method")
        
        smoothed_geoms = []
        
        for geom in gdf.geometry:
            if method == 'gaussian':
                smoothed = self._gaussian_smooth(geom)
            elif method == 'chaikin':
                smoothed = self._chaikin_smooth(geom)
            elif method == 'rdp':
                smoothed = self._rdp_smooth(geom)
            else:
                smoothed = geom
            
            smoothed_geoms.append(smoothed)
        
        gdf_smoothed = gdf.copy()
        gdf_smoothed.geometry = smoothed_geoms
        
        return gdf_smoothed
    
    def _gaussian_smooth(self, line: LineString, sigma: float = 1.0) -> LineString:
        """Apply Gaussian smoothing to line."""
        from scipy.ndimage import gaussian_filter1d
        
        coords = np.array(line.coords)
        smoothed = gaussian_filter1d(coords, sigma=sigma, axis=0)
        
        return LineString(smoothed)
    
    def _chaikin_smooth(self, line: LineString, iterations: int = 2) -> LineString:
        """Apply Chaikin smoothing algorithm."""
        coords = np.array(line.coords)
        
        for _ in range(iterations):
            new_coords = []
            for i in range(len(coords) - 1):
                p1 = coords[i]
                p2 = coords[i + 1]
                new_coords.append(p1 + 0.25 * (p2 - p1))
                new_coords.append(p1 + 0.75 * (p2 - p1))
            
            new_coords.append(coords[-1])
            coords = np.array(new_coords)
        
        return LineString(coords)
    
    def _rdp_smooth(self, line: LineString, epsilon: float = 0.01) -> LineString:
        """Apply Ramer-Douglas-Peucker simplification."""
        return line.simplify(epsilon)
    
    def extract_shoreline_points(self, gdf: gpd.GeoDataFrame, 
                                spacing: float = 30.0) -> gpd.GeoDataFrame:
        """
        Extract discrete points along shoreline at regular intervals.
        
        Args:
            gdf: GeoDataFrame with shoreline geometries
            spacing: Distance between points in meters
            
        Returns:
            GeoDataFrame with point geometries
        """
        self.logger.info(f"Extracting shoreline points at {spacing}m spacing")
        
        points = []
        
        for geom in gdf.geometry:
            # Extract points at regular intervals
            current_distance = 0
            while current_distance < geom.length:
                point = geom.interpolate(current_distance)
                points.append({'distance': current_distance, 'geometry': point})
                current_distance += spacing
        
        gdf_points = gpd.GeoDataFrame(points, crs=gdf.crs)
        self.logger.info(f"Extracted {len(points)} shoreline points")
        
        return gdf_points
    
    def validate_shoreline(self, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Validate detected shoreline geometries.
        
        Args:
            gdf: GeoDataFrame with shoreline geometries
            
        Returns:
            DataFrame with validation results
        """
        self.logger.info("Validating shoreline geometries")
        
        validation_results = []
        
        for idx, row in gdf.iterrows():
            geom = row.geometry
            
            result = {
                'id': idx,
                'is_valid': geom.is_valid,
                'length': geom.length,
                'num_coords': len(geom.coords),
                'bounds': geom.bounds,
                'geom_type': geom.geom_type
            }
            
            validation_results.append(result)
        
        df_validation = pd.DataFrame(validation_results)
        self.logger.info(f"Validation complete: {df_validation['is_valid'].sum()}/{len(df_validation)} valid")
        
        return df_validation
    
    def generate_report(self, gdf: gpd.GeoDataFrame) -> dict:
        """Generate shoreline detection report."""
        report = {
            'total_features': len(gdf),
            'total_length': gdf.geometry.length.sum(),
            'mean_length': gdf.geometry.length.mean(),
            'bounds': gdf.total_bounds,
            'timestamp': pd.Timestamp.now()
        }
        
        self.logger.info(f"Shoreline detection report: {report}")
        return report


if __name__ == "__main__":
    # Example usage
    detector = ShorelineDetector()
    print("Shoreline detector initialized")
