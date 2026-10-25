import unittest
import os
import sys
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predict import predict_point, predict_by_coordinates, CROP_CLASSES

class TestSoilPredictionEngine(unittest.TestCase):
    
    def setUp(self):
        self.standard_input = {
            'soil_pH': 6.20, 'soil_EC_dS_m': 0.45, 'soil_organic_carbon_pct': 0.65,
            'available_N_kg_ha': 220.0, 'available_P_kg_ha': 24.5, 'available_K_kg_ha': 160.0,
            'available_S_mg_kg': 18.0, 'Zn_mg_kg': 0.75, 'Fe_mg_kg': 18.5, 'Mn_mg_kg': 8.2,
            'Cu_mg_kg': 0.90, 'B_mg_kg': 0.48, 'annual_mean_temperature_C': 25.2,
            'annual_rainfall_mm': 1580.0, 'relative_humidity_pct': 75.0,
            'latitude_approx': 23.24, 'longitude_approx': 87.86
        }
        
    def test_standard_prediction(self):
        res = predict_point(self.standard_input, season='Kharif', soil_zone='Alluvial')
        self.assertIn('Soil_Fertility_Index', res)
        self.assertIn('Fertility_Classification', res)
        self.assertIn('Top_3_Suitable_Crops', res)
        self.assertEqual(len(res['Top_3_Suitable_Crops']), 3)
        self.assertIn('Recommended_Fertilizer_kg_ha', res)
        self.assertFalse(res['OOD_Uncertainty_Warning'])
        
    def test_top_crops_validity(self):
        res = predict_point(self.standard_input)
        for crop_info in res['Top_3_Suitable_Crops']:
            self.assertIn(crop_info['Crop'], CROP_CLASSES)
            self.assertGreaterEqual(crop_info['Confidence_Pct'], 0.0)
            self.assertLessEqual(crop_info['Confidence_Pct'], 100.0)
            
    def test_extreme_acidic_ood(self):
        acidic_input = self.standard_input.copy()
        acidic_input['soil_pH'] = 3.2 # Extreme acidic OOD
        res = predict_point(acidic_input)
        self.assertTrue(res['OOD_Uncertainty_Warning'])
        
    def test_invalid_negative_ph_exception(self):
        bad_input = self.standard_input.copy()
        bad_input['soil_pH'] = -1.5
        with self.assertRaises(ValueError):
            predict_point(bad_input)
            
    def test_geospatial_lookup(self):
        geo_res = predict_by_coordinates(23.24, 87.86, season='Kharif')
        self.assertIn('Nearest_District', geo_res)
        self.assertIn('Top_3_Suitable_Crops', geo_res)
        self.assertIn('Soil_Fertility_Index', geo_res)

if __name__ == '__main__':
    unittest.main()
