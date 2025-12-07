import './WelcomeScreen.css'

function WelcomeScreen() {
  return (
    <div className="welcome-screen">
      <div className="welcome-content">
        <div className="welcome-icon">
          <svg viewBox="0 0 64 64" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="32" cy="32" r="28"/>
            <path d="M20 16v32M32 12v40M44 20v24" strokeWidth="3" strokeLinecap="round"/>
          </svg>
        </div>
        
        <h2>WellLog Viewer</h2>
        <p>Professional LAS File Visualization</p>
        
        <div className="welcome-instructions">
          <div className="instruction-item">
            <span className="instruction-number">1</span>
            <span>Click "Open LAS File" in the Explorer panel</span>
          </div>
          <div className="instruction-item">
            <span className="instruction-number">2</span>
            <span>Select a .LAS file from your computer</span>
          </div>
          <div className="instruction-item">
            <span className="instruction-number">3</span>
            <span>View and customize your well log display</span>
          </div>
        </div>
        
        <div className="welcome-features">
          <h4>Features</h4>
          <ul>
            <li>✓ Gamma Ray, Resistivity, Density-Neutron tracks</li>
            <li>✓ Adjustable depth scale (1:200, 1:500, 1:1000)</li>
            <li>✓ Curve visibility controls</li>
            <li>✓ Export to PNG, PDF, LAS</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default WelcomeScreen
