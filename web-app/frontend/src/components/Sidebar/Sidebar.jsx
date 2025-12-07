import { useRef, useState } from 'react'
import './Sidebar.css'

function Sidebar({ 
  wellData, 
  settings, 
  onSettingsChange, 
  onFileUpload, 
  onDepthRangeChange,
  loading,
  curveData
}) {
  const fileInputRef = useRef(null)
  const [expandedSections, setExpandedSections] = useState({
    headers: true,
    logs: true
  })
  const [selectedCurve, setSelectedCurve] = useState(null)
  
  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      onFileUpload(file)
    }
  }
  
  const handleBrowseClick = () => {
    fileInputRef.current?.click()
  }
  
  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }
  
  const getCurveIcon = (curveType) => {
    const icons = {
      GR: '📊',
      RES_DEEP: '〰️',
      RES_MED: '〰️',
      RES_SHAL: '〰️',
      DENS: '📈',
      NEUT: '📉'
    }
    return icons[curveType] || '📊'
  }
  
  const getCurveColorClass = (curveType) => {
    const colorMap = {
      GR: 'curve-gr',
      RES_DEEP: 'curve-res',
      DENS: 'curve-dens',
      NEUT: 'curve-neut'
    }
    return colorMap[curveType] || 'curve'
  }
  
  // Build curve tree from mapping
  const buildCurveTree = () => {
    if (!wellData?.mapping) return []
    
    const mapping = wellData.mapping
    const curves = []
    
    if (mapping.GR) curves.push({ key: 'GR', name: 'Gamma Ray', mnemonic: mapping.GR, type: 'GR' })
    if (mapping.RES_DEEP) curves.push({ key: 'RES_DEEP', name: 'Deep Resistivity', mnemonic: mapping.RES_DEEP, type: 'RES_DEEP' })
    if (mapping.RES_MED) curves.push({ key: 'RES_MED', name: 'Medium Resistivity', mnemonic: mapping.RES_MED, type: 'RES_MED' })
    if (mapping.RES_SHAL) curves.push({ key: 'RES_SHAL', name: 'Shallow Resistivity', mnemonic: mapping.RES_SHAL, type: 'RES_SHAL' })
    if (mapping.DENS) curves.push({ key: 'DENS', name: 'Density (RHOB)', mnemonic: mapping.DENS, type: 'DENS' })
    if (mapping.NEUT) curves.push({ key: 'NEUT', name: 'Neutron (NPHI)', mnemonic: mapping.NEUT, type: 'NEUT' })
    
    return curves
  }
  
  const curves = buildCurveTree()
  
  return (
    <aside className="explorer">
      {/* Header */}
      <div className="explorer-header">
        <svg className="explorer-header-icon" viewBox="0 0 16 16" fill="currentColor">
          <path d="M1 2a1 1 0 011-1h3a1 1 0 011 1v3a1 1 0 01-1 1H2a1 1 0 01-1-1V2zm5 0a1 1 0 011-1h3a1 1 0 011 1v3a1 1 0 01-1 1H7a1 1 0 01-1-1V2zm5 0a1 1 0 011-1h3a1 1 0 011 1v3a1 1 0 01-1 1h-3a1 1 0 01-1-1V2zM1 7a1 1 0 011-1h3a1 1 0 011 1v3a1 1 0 01-1 1H2a1 1 0 01-1-1V7z"/>
        </svg>
        Explorer
      </div>
      
      {/* Content */}
      <div className="explorer-content">
        {/* Headers/Trailers Section */}
        <div className="explorer-section">
          <div 
            className="explorer-section-header"
            onClick={() => toggleSection('headers')}
          >
            <span>{expandedSections.headers ? '▼' : '▶'}</span>
            <span>📋 Item: Headers/Trailers</span>
          </div>
          {expandedSections.headers && (
            <div className="explorer-section-content">
              <div className="tree-view">
                <div className="tree-item">
                  <div className="tree-item-row">
                    <span className="tree-icon folder">📁</span>
                    <span className="tree-label">Default</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
        
        {/* Logs Section */}
        <div className="explorer-section">
          <div 
            className="explorer-section-header"
            onClick={() => toggleSection('logs')}
          >
            <span>{expandedSections.logs ? '▼' : '▶'}</span>
            <span>📊 Item: Logs</span>
          </div>
          {expandedSections.logs && (
            <div className="explorer-section-content">
              <div className="tree-view">
                {curves.length === 0 ? (
                  <div className="tree-item">
                    <div className="tree-item-row" style={{ color: 'var(--color-text-muted)', fontStyle: 'italic' }}>
                      No curves loaded
                    </div>
                  </div>
                ) : (
                  curves.map(curve => (
                    <div className="tree-item" key={curve.key}>
                      <div 
                        className={`tree-item-row ${selectedCurve === curve.key ? 'selected' : ''}`}
                        onClick={() => setSelectedCurve(curve.key)}
                      >
                        <span className={`tree-icon ${getCurveColorClass(curve.type)}`}>
                          {getCurveIcon(curve.type)}
                        </span>
                        <span className="tree-label">{curve.mnemonic}</span>
                        <input 
                          type="checkbox" 
                          className="tree-checkbox"
                          checked={settings.trackSettings?.[curve.key.toLowerCase()]?.visible !== false}
                          onChange={(e) => {
                            e.stopPropagation()
                            // Toggle visibility would go here
                          }}
                        />
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Upload Section at Bottom */}
      <div className="explorer-upload">
        <input
          ref={fileInputRef}
          type="file"
          accept=".las,.LAS"
          onChange={handleFileChange}
          className="hidden-input"
        />
        <button className="upload-btn" onClick={handleBrowseClick} disabled={loading}>
          {loading ? (
            <>
              <div className="spinner-small"></div>
              Processing...
            </>
          ) : (
            <>
              <span>📂</span>
              Open LAS File
            </>
          )}
        </button>
        
        {wellData && (
          <div className="file-info-compact">
            <div className="file-info-name">{wellData.filename}</div>
            <div className="file-info-meta">
              {wellData.depth_range.min.toFixed(1)} - {wellData.depth_range.max.toFixed(1)} {wellData.depth_unit}
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}

export default Sidebar
