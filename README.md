# WellLog Viewer Pro

Professional LAS file visualization application with industry-standard well log display.

## Project Structure

```
logai/
├── shared/              # Shared Python utilities
│   ├── las_parser.py    # LAS file loading and parsing
│   ├── curve_mapping.py # Automatic curve detection
│   └── data_processing.py # Data processing utilities
│
├── streamlit-app/       # Streamlit version
│   ├── app.py          # Main Streamlit application
│   ├── plotting.py     # Matplotlib-based plotting
│   └── run.sh          # Run script
│
└── web-app/            # Flask + React version
    ├── backend/        # Flask REST API
    │   └── app.py      # API endpoints
    └── frontend/       # React application
        └── src/        # React components
```

## Quick Start

### Streamlit Version
```bash
cd streamlit-app
./run.sh
# Open http://localhost:8501
```

### Flask + React Version
```bash
cd web-app
./run.sh
# Open http://localhost:3000
```

## Features

### Implemented ✅
- LAS file upload and parsing
- Automatic curve detection (GR, Resistivity, Density, Neutron)
- Professional Techlog-style visualization
- Multiple tracks with industry-standard colors
- Depth scale options (1:200, 1:500, 1:1000)
- Curve smoothing
- GR sand shading (optional)
- Density-Neutron crossover fill
- Missing data handling (gaps, not lines)
- Export to PNG, PDF, LAS
- Well header metadata display
- Depth range selection

### Track Configuration
| Track | Curve | Color | Scale | Range |
|-------|-------|-------|-------|-------|
| 1 | Gamma Ray (GR) | Green | Linear | 25-125 API |
| 2 | Resistivity (RILD) | Red | Logarithmic | 0.2-200 ohm.m |
| 3 | Density (RHOB) | Red | Linear | 1.85-2.85 g/cc |
| 3 | Neutron (NPHI) | Blue | Linear | -0.15-0.45 v/v |

### Future Placeholders 🚀
- Multi-well comparison (side-by-side)
- Lithology track
- Zonal shading and markers
- Formation tops

## API Endpoints (Flask Backend)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/upload | Upload LAS file |
| GET | /api/wells | List uploaded wells |
| GET | /api/wells/{id} | Get well metadata |
| GET | /api/wells/{id}/curves | Get curve data |
| GET | /api/wells/{id}/export/las | Export as LAS |
| DELETE | /api/wells/{id} | Delete well |

## Technology Stack

### Streamlit Version
- Python 3.8+
- Streamlit
- Matplotlib
- lasio
- pandas

### Flask + React Version
- **Backend**: Flask, Flask-CORS, lasio, pandas
- **Frontend**: React 18, Vite, D3.js

## License

MIT License
