import './WellHeader.css'

function WellHeader({ headerInfo, depthUnit }) {
  if (!headerInfo) return null
  
  return (
    <div className="well-header">
      <div className="well-header-main">
        <span className="well-name">WELL: {headerInfo.WELL || 'UNKNOWN'}</span>
      </div>
      
      <div className="well-header-details">
        <div className="header-field">
          <span className="field-label">FIELD:</span>
          <span className="field-value">{headerInfo.FIELD || 'N/A'}</span>
        </div>
        
        <div className="header-field">
          <span className="field-label">LOCATION:</span>
          <span className="field-value">{headerInfo.LOC || 'N/A'}</span>
        </div>
        
        <div className="header-field">
          <span className="field-label">OPERATOR:</span>
          <span className="field-value">{headerInfo.COMP || 'N/A'}</span>
        </div>
        
        <div className="header-field">
          <span className="field-label">DEPTH:</span>
          <span className="field-value">
            {headerInfo.STRT} – {headerInfo.STOP} {depthUnit}
          </span>
        </div>
        
        <div className="header-field">
          <span className="field-label">STEP:</span>
          <span className="field-value">{headerInfo.STEP} {depthUnit}</span>
        </div>
        
        {headerInfo.DATE && (
          <div className="header-field">
            <span className="field-label">DATE:</span>
            <span className="field-value">{headerInfo.DATE}</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default WellHeader
