import { useState } from 'react'
import { exportLas } from '../../services/api'
import './ExportPanel.css'

function ExportPanel({ wellData, settings }) {
  const [exporting, setExporting] = useState(null)
  
  if (!wellData) return null
  
  const handleExportPng = async () => {
    setExporting('png')
    try {
      // Get the SVG element
      const svg = document.querySelector('.log-viewer-canvas svg')
      if (!svg) throw new Error('No chart found')
      
      // Convert SVG to canvas and download
      const svgData = new XMLSerializer().serializeToString(svg)
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      const img = new Image()
      
      canvas.width = svg.getAttribute('width') * 2
      canvas.height = svg.getAttribute('height') * 2
      ctx.scale(2, 2)
      ctx.fillStyle = 'white'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      
      img.onload = () => {
        ctx.drawImage(img, 0, 0)
        const link = document.createElement('a')
        link.download = `${wellData.header.WELL}_log.png`
        link.href = canvas.toDataURL('image/png')
        link.click()
        setExporting(null)
      }
      
      img.src = 'data:image/svg+xml;base64,' + btoa(unescape(encodeURIComponent(svgData)))
    } catch (err) {
      console.error('PNG export error:', err)
      setExporting(null)
    }
  }
  
  const handleExportPdf = async () => {
    setExporting('pdf')
    // For PDF, we'd typically use a library like jsPDF
    // For now, just trigger print dialog
    try {
      window.print()
    } finally {
      setExporting(null)
    }
  }
  
  const handleExportLas = async () => {
    setExporting('las')
    try {
      const blob = await exportLas(
        wellData.well_id,
        settings.depthRange.start,
        settings.depthRange.end
      )
      
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.download = `${wellData.header.WELL}_export.las`
      link.href = url
      link.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      console.error('LAS export error:', err)
    } finally {
      setExporting(null)
    }
  }
  
  return (
    <div className="export-panel">
      <h3>💾 Export Options</h3>
      
      <div className="export-buttons">
        <button 
          className="export-btn"
          onClick={handleExportPng}
          disabled={exporting === 'png'}
        >
          {exporting === 'png' ? (
            <span className="spinner-small"></span>
          ) : (
            <span className="btn-icon">📷</span>
          )}
          Download PNG
        </button>
        
        <button 
          className="export-btn"
          onClick={handleExportPdf}
          disabled={exporting === 'pdf'}
        >
          {exporting === 'pdf' ? (
            <span className="spinner-small"></span>
          ) : (
            <span className="btn-icon">📄</span>
          )}
          Download PDF
        </button>
        
        <button 
          className="export-btn"
          onClick={handleExportLas}
          disabled={exporting === 'las'}
        >
          {exporting === 'las' ? (
            <span className="spinner-small"></span>
          ) : (
            <span className="btn-icon">📋</span>
          )}
          Export LAS
        </button>
      </div>
    </div>
  )
}

export default ExportPanel
