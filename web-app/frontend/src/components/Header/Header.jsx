import { useState } from 'react'
import './Header.css'

function Header({ wellData, onExportPng, onExportPdf, settings, onSettingsChange }) {
  const [activeMenu, setActiveMenu] = useState(null)
  
  const menuItems = ['File', 'Edit', 'View', 'Process', 'Tools', 'Window', 'Help']
  
  const handleScaleChange = (e) => {
    if (onSettingsChange) {
      onSettingsChange({ scale: Number(e.target.value) })
    }
  }
  
  return (
    <header className="header">
      {/* Title Bar */}
      <div className="title-bar">
        <div className="title-bar-left">
          <div className="title-bar-icon">
            <svg viewBox="0 0 16 16" fill="currentColor">
              <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.5"/>
              <path d="M5 4v8M8 3v10M11 5v6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </div>
          <span className="title-bar-text">
            WellLog Viewer - {wellData?.filename || 'No file loaded'}
          </span>
        </div>
        <div className="title-bar-controls">
          <button className="title-bar-btn">─</button>
          <button className="title-bar-btn">□</button>
          <button className="title-bar-btn close">✕</button>
        </div>
      </div>
      
      {/* Menu Bar */}
      <div className="menu-bar">
        {menuItems.map(item => (
          <button 
            key={item} 
            className="menu-item"
            onMouseEnter={() => activeMenu && setActiveMenu(item)}
            onClick={() => setActiveMenu(activeMenu === item ? null : item)}
          >
            {item}
          </button>
        ))}
      </div>
      
      {/* Toolbar */}
      <div className="toolbar">
        {/* File operations */}
        <div className="toolbar-group">
          <button className="toolbar-btn" title="New">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="3" y="2" width="10" height="12" rx="1"/>
              <path d="M5 5h6M5 8h6M5 11h4"/>
            </svg>
          </button>
          <button className="toolbar-btn" title="Open">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2 5v8a1 1 0 001 1h10a1 1 0 001-1V7a1 1 0 00-1-1H8L6 4H3a1 1 0 00-1 1z"/>
            </svg>
          </button>
          <button className="toolbar-btn" title="Save">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M3 2h8l2 2v9a1 1 0 01-1 1H3a1 1 0 01-1-1V3a1 1 0 011-1z"/>
              <rect x="5" y="2" width="6" height="4"/>
              <rect x="4" y="9" width="8" height="4"/>
            </svg>
          </button>
        </div>
        
        {/* Print/Export */}
        <div className="toolbar-group">
          <button className="toolbar-btn" title="Print">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="4" y="2" width="8" height="4"/>
              <rect x="2" y="6" width="12" height="6" rx="1"/>
              <rect x="4" y="10" width="8" height="4"/>
            </svg>
          </button>
          <button 
            className="toolbar-btn" 
            title="Export PNG"
            onClick={onExportPng}
          >
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <rect x="2" y="2" width="12" height="9" rx="1"/>
              <path d="M5 14h6M8 11v3"/>
            </svg>
          </button>
          <button 
            className="toolbar-btn" 
            title="Export PDF"
            onClick={onExportPdf}
          >
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.2">
              <rect x="3" y="1" width="10" height="14" rx="1"/>
              <text x="5" y="11" fontSize="5" fill="currentColor" stroke="none">PDF</text>
            </svg>
          </button>
        </div>
        
        {/* View controls */}
        <div className="toolbar-group">
          <button className="toolbar-btn" title="Zoom In">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="7" cy="7" r="4"/>
              <path d="M10 10l3 3M5 7h4M7 5v4"/>
            </svg>
          </button>
          <button className="toolbar-btn" title="Zoom Out">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="7" cy="7" r="4"/>
              <path d="M10 10l3 3M5 7h4"/>
            </svg>
          </button>
          <div className="zoom-control">
            <select 
              value={settings?.scale || 500} 
              onChange={handleScaleChange}
              title="Depth Scale"
            >
              <option value={200}>1:200</option>
              <option value={500}>1:500</option>
              <option value={1000}>1:1000</option>
              <option value={2000}>1:2000</option>
            </select>
          </div>
        </div>
        
        {/* Tools */}
        <div className="toolbar-group">
          <button className="toolbar-btn" title="Cursor">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M3 3l4 10 2-4 4-2z"/>
            </svg>
          </button>
          <button className="toolbar-btn" title="Pan">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M8 2v12M2 8h12M8 2l2 2M8 2l-2 2M8 14l2-2M8 14l-2-2M2 8l2 2M2 8l2-2M14 8l-2 2M14 8l-2-2"/>
            </svg>
          </button>
          <button className="toolbar-btn disabled" title="Measure (Coming soon)">
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2 14l12-12M2 14l3-1-2-2z"/>
              <path d="M5 11l1-1M7 9l1-1M9 7l1-1M11 5l1-1"/>
            </svg>
          </button>
        </div>
        
        <div className="scale-display">
          Scale 1:{settings?.scale || 500}
        </div>
      </div>
    </header>
  )
}

export default Header
