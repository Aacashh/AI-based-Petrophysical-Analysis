# Streamlit Application Architecture

## Overview

This document describes the architecture of the Petrophysical Analysis Streamlit application and provides guidance for porting it to a Flask backend.

---

## Project Structure

```
streamlit-app/
├── app.py                          # Main entry point (landing page)
├── pages/
│   └── 04_Petrophysical_Analysis.py  # Main analysis page (4,044 lines)
├── ml_components/                  # ML/AI processing modules (5,045 lines)
│   ├── __init__.py
│   ├── outlier_detection.py        # Isolation Forest, LOF, ABOD algorithms
│   ├── bayesian_optimizer.py       # Hyperparameter optimization
│   ├── lstm_alignment.py           # Deep learning alignment (PyTorch)
│   ├── cnn_pattern_matcher.py      # Siamese CNN for pattern matching
│   ├── depth_alignment.py          # Multi-method depth alignment
│   ├── noise_removal.py            # Tool startup noise detection
│   ├── rock_classification.py      # Unsupervised clustering
│   ├── uncertainty.py              # Bayesian uncertainty quantification
│   └── visualizations.py           # Plotly/HTML visualization components
├── plotting.py                     # Professional log visualization (2,654 lines)
└── .streamlit/config.toml          # Streamlit configuration
```

---

## External Dependencies

### Shared Modules (Located in parent directory)

The application heavily depends on shared modules from `../shared/`:

```python
from shared.las_parser import (
    load_las,
    extract_header_info,
    detect_depth_units,
    get_available_curves
)

from shared.curve_mapping import (
    get_curve_mapping,
    get_curve_mapping_advanced,
    get_identification_summary
)

from shared.data_processing import (
    process_data,
    export_to_las
)

from shared.splicing import (
    group_files_by_well,
    WellGroupResult,
    splice_logs
)

from shared.ascii_parser import (
    load_ascii_file,
    AsciiFileMetadata,
    DepthDetectionResult,
    create_pseudo_las_from_ascii,
    validate_ascii_data
)

from shared.curve_identifier import (
    CurveIdentifier,
    identify_curves_layered,
    format_identification_report,
    CurveType,
    FileType,
    ToolContext,
    IdentificationReport
)
```

**These modules handle:**
- LAS/DLIS file parsing
- Curve identification using 12-layer methodology
- Data processing and export
- Log splicing algorithms
- ASCII file import

---

## Application Architecture

### 1. Entry Point (app.py)

**Responsibilities:**
- Landing page with welcome message
- Feature overview
- Quick start guide
- CSS styling

**Lines of Code:** 178 lines
**Complexity:** Low (mostly static content)

---

### 2. Main Analysis Page (04_Petrophysical_Analysis.py)

**Responsibilities:**
- File upload handling (folder, single file, ASCII, multiple files)
- Data loading and preprocessing
- 5 major workflow tabs:
  1. Outlier Detection
  2. Noise Removal
  3. Log Splicing
  4. Depth Alignment
  5. Rock Classification
- Professional log visualization
- Curve identification UI
- Export functionality

**Lines of Code:** 4,044 lines
**Complexity:** Very High (monolithic)

#### Helper Functions (Only 4 defined):

1. **`scan_uploaded_files_for_field_well(uploaded_files)`** (Line 377)
   - Scans uploaded LAS files and extracts Field/Well metadata
   - Returns organized dict with file info by field and well
   - Used for folder browsing mode

2. **`get_available_feature_columns(df, mapping)`** (Line 590)
   - Extracts available feature columns from dataframe
   - Excludes DEPTH and non-numeric columns
   - Returns list of curve names suitable for analysis

3. **`_identify_curves_from_mnemonics(curve_names, df=None)`** (Line 599)
   - Legacy curve identification helper
   - Maps curve mnemonics to standard curve types
   - Fallback when advanced identification unavailable

4. **`display_file_info(header, df, unit, col)`** (Line 714)
   - Renders file metadata in UI column
   - Displays Well, Field, Location, depth range, sample count
   - Pure UI rendering function

---

## Data Flow

### Single File Workflow

```
User Uploads File
    ↓
load_las() / load_ascii_file()
    ↓
extract_header_info()
detect_depth_units()
    ↓
get_curve_mapping_advanced()  [12-layer identification]
    ↓
process_data()  [Create DataFrame with standardized columns]
    ↓
User Selects Workflow Tab
    ↓
Tab 1: detect_outliers_*() → clean_outliers()
Tab 2: detect_tool_startup_noise() → remove_noise()
Tab 3: N/A (requires multiple files)
Tab 4: N/A (requires multiple files)
Tab 5: find_optimal_clusters() → classify_rock_types()
    ↓
create_professional_log_display()
    ↓
export_to_las()  [Optional]
```

### Multi-File Workflow

```
User Uploads Multiple Files
    ↓
scan_uploaded_files_for_field_well()
group_files_by_well()
    ↓
[For each file: load → identify curves → process]
    ↓
User Selects Workflow Tab
    ↓
Tab 3 (Splicing):
    detect_best_overlap() → splice_logs_dtw()
    optimize_splicing_params() [Bayesian optimization]

Tab 4 (Alignment):
    align_by_correlation() [Classical]
    OR
    train_siamese_network() → predict_alignment() [ML]
    ↓
create_professional_log_display() [Multi-file overlay]
    ↓
export_to_las()
```

---

## ML Components Architecture

### Design Principles

All ML components follow these principles:
1. **Zero Streamlit dependencies** - Can be imported in any Python project
2. **Dataclass result containers** - Structured, typed outputs
3. **Clear function signatures** - Explicit inputs and outputs
4. **Modular and testable** - Each component can be tested independently

### Component Categories

#### Tier 1: Fast ML (Sub-second inference)

- **outlier_detection.py**
  - `detect_outliers_isolation_forest()` → `OutlierDetectionResult`
  - `detect_outliers_lof()` → `OutlierDetectionResult`
  - `detect_outliers_abod()` → `OutlierDetectionResult`
  - `detect_outliers_ensemble()` → `OutlierDetectionResult`
  - `clean_outliers()` → `pd.DataFrame`

- **noise_removal.py**
  - `detect_tool_startup_noise()` → `NoiseDetectionResult`
  - `remove_noise()` → `NoiseRemovalResult`

- **bayesian_optimizer.py**
  - `optimize_splicing_params()` → `OptimizationResult`

#### Tier 2: Deep Learning (Requires training)

- **lstm_alignment.py**
  - `LightweightLSTMAligner(nn.Module)` - PyTorch model
  - `train_lstm_aligner()` → `LSTMTrainingResult`
  - `predict_alignment()` → `AlignmentPrediction`

- **cnn_pattern_matcher.py**
  - `SiameseCNN(nn.Module)` - 1D CNN architecture
  - `train_pattern_matcher()` → `PatternMatchResult`
  - `compute_similarity_map()` → `SimilarityResult`

- **depth_alignment.py**
  - `align_by_correlation()` → `AlignmentResult`
  - Integration with Siamese network → `SiameseAlignmentResult`

- **rock_classification.py**
  - `find_optimal_clusters()` → `RockClassificationResult`
  - K-means and GMM implementations
  - Petrophysical interpretation logic

#### Tier 3: Uncertainty & Visualization

- **uncertainty.py**
  - `predict_with_uncertainty()` → `UncertaintyResult`
  - Monte Carlo Dropout for confidence intervals

- **visualizations.py**
  - `render_lstm_architecture()` → HTML/SVG diagram
  - `render_cnn_architecture()` → Network visualization
  - `create_metrics_dashboard()` → Plotly figures
  - `create_comparison_plot()` → Before/after visualizations

---

## State Management

### Streamlit Session State Usage

The application uses `st.session_state` extensively for maintaining state across user interactions:

```python
# Common state variables used:
st.session_state['outlier_result'] = result
st.session_state['outlier_df'] = df
st.session_state['noise_result'] = result
st.session_state['cleaned_df'] = cleaned_df
st.session_state['ascii_import_complete'] = True
st.session_state['selected_curves'] = curves
```

**Flask Migration Note:** Flask requires explicit session management or database storage. Consider:
- Redis for session storage
- PostgreSQL for persistent data
- Background job queues (Celery) for long-running ML tasks

---

## Critical Issues for Flask Migration

### 1. Monolithic Page Structure

**Current:** Single 4,044-line file with all workflows
**Problem:** Business logic tightly coupled with UI rendering
**Solution Required:** Extract into separate service classes

### 2. No Service Layer

**Current Pattern:**
```python
if st.button("Run Outlier Detection"):
    # Business logic directly in button handler
    result = detect_outliers_isolation_forest(df, curves, contamination)
    # Immediately render results
    st.metric("Outliers", result.num_anomalies)
```

**Required Pattern for Flask:**
```python
# Route handler (routes/outlier_detection.py)
@blueprint.route('/api/outlier-detection', methods=['POST'])
def detect_outliers():
    data = request.get_json()
    result = OutlierService.detect(data)
    return jsonify(result.to_dict())

# Service layer (services/outlier_service.py)
class OutlierService:
    @staticmethod
    def detect(data):
        df = pd.DataFrame(data['data'])
        return detect_outliers_isolation_forest(
            df, data['curves'], data['contamination']
        )
```

### 3. File Upload Complexity

**Current:** Multiple upload modes with complex branching
**Problem:** Files processed synchronously with spinner feedback
**Solution Required:** Async upload handling with background jobs

### 4. Embedded CSS (360+ lines)

**Current:** All CSS embedded as Python strings
**Problem:** Hard to maintain and test
**Solution Required:** Extract to `static/css/` files

### 5. Visualization Dependencies

**Current:** matplotlib figures passed to `st.pyplot()`
**Problem:** Need HTTP response handling
**Solution Required:** Save figures to BytesIO, return as image responses

---

## Recommended Refactoring Steps

### Phase 1: Service Layer Extraction (High Priority)

Create service classes to decouple business logic from UI:

```python
# services/outlier_service.py
from dataclasses import dataclass
from ml_components.outlier_detection import detect_outliers_isolation_forest
import logging

logger = logging.getLogger(__name__)

class OutlierService:
    """Business logic for outlier detection workflows."""

    @staticmethod
    def detect(df, curves, contamination, method='isolation_forest'):
        """
        Detect outliers in well log data.

        Args:
            df: DataFrame with well log data
            curves: List of curve names to analyze
            contamination: Expected outlier percentage (0.01-0.20)
            method: Detection method ('isolation_forest', 'lof', 'abod', 'ensemble')

        Returns:
            OutlierDetectionResult with num_anomalies, indices, confidence

        Raises:
            ValueError: If curves not in df or contamination out of range
        """
        logger.info(f"Running {method} outlier detection on {len(curves)} curves")

        # Validate inputs
        if not all(c in df.columns for c in curves):
            missing = [c for c in curves if c not in df.columns]
            raise ValueError(f"Curves not found in data: {missing}")

        if not 0.01 <= contamination <= 0.20:
            raise ValueError("Contamination must be between 0.01 and 0.20")

        # Run detection
        if method == 'isolation_forest':
            result = detect_outliers_isolation_forest(df, curves, contamination)
        elif method == 'lof':
            result = detect_outliers_lof(df, curves, contamination)
        elif method == 'abod':
            result = detect_outliers_abod(df, curves, contamination)
        else:
            result = detect_outliers_ensemble(df, curves, contamination)

        logger.info(f"Detected {result.num_anomalies} outliers ({result.contamination_actual:.1%})")
        return result

    @staticmethod
    def clean(df, result, method='interpolate'):
        """Clean detected outliers from data."""
        logger.info(f"Cleaning {result.num_anomalies} outliers using {method}")
        return clean_outliers(df, result, method=method)
```

Repeat this pattern for:
- `NoiseRemovalService`
- `SplicingService`
- `AlignmentService`
- `ClassificationService`

### Phase 2: API Route Structure (High Priority)

Organize Flask routes by workflow:

```
backend/
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── outlier_detection.py
│   │   ├── noise_removal.py
│   │   ├── splicing.py
│   │   ├── alignment.py
│   │   └── classification.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── request_models.py
│   │   └── response_models.py
│   └── middleware/
│       ├── error_handler.py
│       └── validation.py
├── services/
│   ├── __init__.py
│   ├── outlier_service.py
│   ├── noise_removal_service.py
│   ├── splicing_service.py
│   ├── alignment_service.py
│   └── classification_service.py
├── ml_components/  # Import from streamlit-app
└── shared/         # Import from parent
```

### Phase 3: Response Standardization (Medium Priority)

All API responses should follow a consistent format:

```python
# Success response
{
    "status": "success",
    "data": {
        "num_outliers": 42,
        "contamination_actual": 0.051,
        "confidence": 0.87,
        "method": "isolation_forest"
    },
    "message": "Successfully detected 42 outliers"
}

# Error response
{
    "status": "error",
    "error": {
        "code": "INVALID_INPUT",
        "message": "Contamination must be between 0.01 and 0.20",
        "details": {
            "provided_value": 0.35,
            "valid_range": [0.01, 0.20]
        }
    }
}
```

### Phase 4: Template Extraction (Medium Priority)

Extract HTML/CSS from Python strings:

```
templates/
├── base.html
├── index.html
├── workflows/
│   ├── outlier_detection.html
│   ├── noise_removal.html
│   ├── splicing.html
│   ├── alignment.html
│   └── classification.html
└── components/
    ├── file_upload.html
    ├── curve_selector.html
    └── results_display.html

static/
├── css/
│   ├── main.css
│   └── components.css
└── js/
    ├── plotly_config.js
    └── file_upload.js
```

### Phase 5: Background Job Processing (High Priority)

ML operations can take several minutes. Flask requires async handling:

```python
# Using Celery for background tasks
from celery import Celery

celery = Celery('logai', broker='redis://localhost:6379')

@celery.task
def run_splicing_task(file_ids, params):
    """Background task for ML-based log splicing."""
    # Load files
    files = FileService.load_multiple(file_ids)

    # Run splicing
    result = SplicingService.ml_splice(files, params)

    # Store result
    ResultStore.save(result)

    return result.task_id

# Flask route
@blueprint.route('/api/splicing/start', methods=['POST'])
def start_splicing():
    data = request.get_json()
    task = run_splicing_task.delay(data['file_ids'], data['params'])
    return jsonify({
        'status': 'processing',
        'task_id': task.id,
        'poll_url': f'/api/splicing/status/{task.id}'
    })

@blueprint.route('/api/splicing/status/<task_id>')
def splicing_status(task_id):
    task = run_splicing_task.AsyncResult(task_id)
    if task.ready():
        return jsonify({
            'status': 'completed',
            'result': task.result
        })
    else:
        return jsonify({
            'status': 'processing',
            'progress': task.info.get('progress', 0)
        })
```

---

## Key Modularity Issues

### Issue 1: Business Logic Mixed with UI

**Problem Location:** Throughout entire `04_Petrophysical_Analysis.py`

**Example (Lines 1434-1450):**
```python
if st.button("Run Outlier Detection"):
    with st.spinner("Detecting outliers..."):
        # Business logic in UI handler
        if outlier_method == 'Isolation Forest':
            result = detect_outliers_isolation_forest(df, selected_features, contamination)

        # Immediate UI rendering
        st.markdown('<div class="feature-header">Detection Results</div>')
        st.metric("Outliers", result.num_anomalies)
```

**Required Separation:**
- **Route Handler:** Parse request, call service, format response
- **Service Layer:** Validate inputs, call ML component, return result
- **ML Component:** Pure algorithm logic (already separated!)
- **Template:** Render results (HTML/Jinja2)

### Issue 2: Procedural Code (3,900+ lines)

**Problem:** Only 4 helper functions for 4,044 lines of code

**Impact:**
- Code duplication across tabs
- Difficult to test individual workflows
- No reusable components for Flask API

**Solution:** Extract functions for each logical operation:
```python
def load_and_process_file(uploaded_file):
    """Load file and return processed DataFrame with metadata."""

def validate_curve_selection(df, curves):
    """Validate that selected curves exist and are numeric."""

def prepare_visualization_data(df, mapping, curves):
    """Prepare data structure for plotting."""
```

### Issue 3: State Management Scattered

**Problem:** Session state access scattered throughout code

**Examples:**
```python
st.session_state['outlier_result'] = result  # Line 1449
st.session_state['cleaned_df'] = cleaned_df  # Line 1586
st.session_state['ascii_import_complete'] = True  # Line 955
```

**Solution:** Centralize state management:
```python
class SessionManager:
    """Manage application session state."""

    @staticmethod
    def store_outlier_result(result, df):
        st.session_state['outlier_result'] = result
        st.session_state['outlier_df'] = df

    @staticmethod
    def get_outlier_result():
        return st.session_state.get('outlier_result')
```

---

## Flask Migration Complexity Estimate

| Component | Lines | Effort | Priority |
|-----------|-------|--------|----------|
| ML Components | 5,045 | Low (reuse as-is) | N/A |
| Shared Modules | ~2,000 | Low (reuse as-is) | N/A |
| Service Layer Extraction | 0 → 800 | High | 1 |
| API Routes | 0 → 500 | High | 1 |
| Request/Response Models | 0 → 300 | Medium | 2 |
| Template Extraction | 4,044 → 1,200 | High | 2 |
| CSS Extraction | 360 → 360 | Low | 3 |
| Background Jobs | 0 → 400 | High | 1 |
| File Upload Handling | ~200 → 300 | Medium | 2 |
| Testing Infrastructure | 0 → 800 | Medium | 3 |

**Total Estimated Effort:** 15-20 days for 1 developer

---

## Strengths (Already Production-Ready)

1. **ML Components:**
   - Clean, modular design
   - No external UI dependencies
   - Dataclass result containers
   - Ready for Flask import

2. **Shared Modules:**
   - Well-tested parsing logic
   - 12-layer curve identification
   - Industry-standard algorithms

3. **Professional Visualization:**
   - `plotting.py` generates matplotlib figures
   - Can be wrapped for Flask with minimal changes
   - Follows industry standards (3-track log display)

---

## Weaknesses (Require Refactoring)

1. **4,044-line Monolithic Page:**
   - All workflows in single file
   - Difficult to maintain and test
   - No separation of concerns

2. **No Service Layer:**
   - Business logic embedded in UI handlers
   - Cannot reuse logic in API endpoints
   - Difficult to test without Streamlit

3. **Synchronous File Processing:**
   - Long-running ML operations block UI
   - No progress tracking for complex workflows
   - Flask needs async/background processing

4. **Embedded Styling:**
   - 360+ lines of CSS in Python strings
   - Difficult to update and maintain
   - Should be extracted to static files

---

## Next Steps for Production Backend

### Immediate Actions (Phase 1):
1. ✅ Remove all emojis from error/success messages
2. ✅ Standardize message formatting
3. Create service layer classes (5 services)
4. Define API request/response schemas
5. Set up Flask project structure

### Short-term (Phase 2):
6. Implement RESTful API endpoints
7. Add background job processing (Celery)
8. Create Jinja2 templates
9. Extract CSS to static files
10. Implement session/database storage

### Medium-term (Phase 3):
11. Add comprehensive logging
12. Implement authentication/authorization
13. Add API documentation (Swagger/OpenAPI)
14. Set up CI/CD pipeline
15. Write integration tests

---

## Testing Strategy

### Current State
- No automated tests in streamlit-app
- Manual testing only

### Required for Production

```python
# Unit tests for services
def test_outlier_service_validation():
    """Test that service validates inputs correctly."""
    with pytest.raises(ValueError):
        OutlierService.detect(df, ['INVALID_CURVE'], 0.05)

# Integration tests for API
def test_outlier_detection_api(client):
    """Test outlier detection endpoint."""
    response = client.post('/api/outlier-detection', json={
        'data': test_data,
        'curves': ['GR', 'RHOB'],
        'contamination': 0.05
    })
    assert response.status_code == 200
    assert 'num_outliers' in response.json['data']

# E2E tests
def test_complete_workflow(client):
    """Test full outlier detection workflow."""
    # Upload file
    # Detect outliers
    # Clean data
    # Export results
```

---

## Performance Considerations

### Current Bottlenecks

1. **Siamese Network Training:** 2-5 minutes for depth alignment
2. **Log Splicing with DTW:** 30-60 seconds for large files
3. **Rock Classification:** 10-20 seconds for optimal cluster search
4. **File Loading:** 5-10 seconds for large DLIS files

### Flask Optimization Strategies

1. **Caching:**
   - Cache parsed LAS files (Redis)
   - Cache curve identification results
   - Cache trained ML models

2. **Async Processing:**
   - Celery for long-running tasks
   - WebSocket for real-time progress updates
   - Task queue with priority levels

3. **Database:**
   - Store processed DataFrames in PostgreSQL
   - Index by well name, field, date
   - Avoid re-processing same files

---

## Dependencies

### Python Packages Required

**Core:**
- streamlit
- pandas
- numpy
- matplotlib

**ML/AI:**
- scikit-learn
- torch (PyTorch)
- pyod (PyOD for ABOD)
- scipy

**File Parsing:**
- lasio (LAS files)
- dlisio (DLIS files)

**Visualization:**
- plotly

**For Flask Migration:**
- flask
- flask-restful
- celery
- redis
- sqlalchemy
- marshmallow (schemas)
- pytest

---

## Conclusion

The current Streamlit application has:
- **Excellent ML component modularity** (ready for Flask)
- **Professional visualization capabilities**
- **Comprehensive curve identification system**

But requires significant refactoring for production:
- **Extract service layer** (800+ lines of new code)
- **Create API routes** (500+ lines)
- **Decompose monolithic page** (split 4,044 lines into 5+ modules)
- **Add background job processing**
- **Implement proper error handling and logging**

The good news: The hardest part (ML algorithms) is already well-architected and reusable.

---

**Document Version:** 1.0
**Last Updated:** 2026-01-30
**Maintainer:** Development Team
