import os
import sys
from utils import load_las, get_curve_mapping, process_data, extract_header_info
from plotting import create_log_plot
import matplotlib.pyplot as plt

def test_pipeline():
    las_path = "/media/vulcan/DATA/Work/deliverables/1055929125.las"
    print(f"Testing with file: {las_path}")
    
    try:
        with open(las_path, 'rb') as f:
            las = load_las(f)
        
        print("LAS loaded successfully.")
        
        mapping = get_curve_mapping(las)
        print("Mapping:", mapping)
        
        df = process_data(las, mapping)
        print("DataFrame shape:", df.shape)
        print("Columns:", df.columns)
        
        header = extract_header_info(las)
        print("Header:", header)
        
        print("Generating plot...")
        settings = {
            'scale_ratio': 500,
            'gr_min': 0, 'gr_max': 150,
            'res_min': 0.2, 'res_max': 2000,
            'dens_min': 1.95, 'dens_max': 2.95,
            'neut_min': -0.15, 'neut_max': 0.45
        }
        fig = create_log_plot(df, mapping, settings)
        
        output_path = "test_plot.png"
        fig.savefig(output_path)
        print(f"Plot saved to {output_path}")
        
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
