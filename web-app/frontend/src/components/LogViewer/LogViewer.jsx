import { useRef, useEffect, useMemo } from 'react'
import * as d3 from 'd3'
import './LogViewer.css'

function LogViewer({ wellData, curveData, settings, loading }) {
  const containerRef = useRef(null)
  const svgRef = useRef(null)
  
  // Track configuration - WellCAD style
  const trackConfig = useMemo(() => ({
    depth: { 
      width: 50,
      title: 'Depth',
      unit: wellData?.depth_unit || 'm'
    },
    gr: { 
      width: 120, 
      title: 'Gamma',
      color: '#00AA00',
      unit: 'GAPI',
      min: settings.trackSettings.gr.min,
      max: settings.trackSettings.gr.max,
      scale: 'linear'
    },
    res: { 
      width: 120, 
      title: 'RPOR',
      subtitle: 'CODE',
      color: '#0000FF',
      unit: 'ohm.m',
      min: settings.trackSettings.res.min,
      max: settings.trackSettings.res.max,
      scale: 'log'
    },
    dn: { 
      width: 120, 
      title: 'MC4F',
      subtitle: 'MC2F',
      densColor: '#00AA00',
      neutColor: '#0000FF',
      densUnit: 'US/M',
      neutUnit: 'US/M',
      densMin: settings.trackSettings.dens.min,
      densMax: settings.trackSettings.dens.max,
      neutMin: settings.trackSettings.neut.min,
      neutMax: settings.trackSettings.neut.max,
    },
    cade: {
      width: 100,
      title: 'CADE',
      color: '#CC0000',
      min: 90,
      max: 130
    }
  }), [settings.trackSettings, wellData?.depth_unit])
  
  // Calculate dimensions
  const dimensions = useMemo(() => {
    const headerHeight = 40
    const totalWidth = trackConfig.depth.width + trackConfig.gr.width + 
                       trackConfig.res.width + trackConfig.dn.width + 
                       trackConfig.cade.width + 20
    
    // Calculate height based on depth range and scale
    const depthRange = (settings.depthRange.end || 0) - (settings.depthRange.start || 0)
    const heightCm = (depthRange * 100) / settings.scale
    const heightPx = Math.max(400, Math.min(heightCm * 37.8, 10000))
    
    return {
      width: totalWidth,
      height: heightPx + headerHeight,
      headerHeight,
      plotHeight: heightPx
    }
  }, [settings, trackConfig])
  
  // Render chart with D3
  useEffect(() => {
    if (!curveData || !svgRef.current) return
    
    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()
    
    const { width, height, headerHeight, plotHeight } = dimensions
    
    // Depth data
    const depth = curveData.depth || []
    const curves = curveData.curves || {}
    const mapping = curveData.mapping || {}
    
    if (depth.length === 0) return
    
    // Depth scale
    const depthMin = d3.min(depth)
    const depthMax = d3.max(depth)
    const yScale = d3.scaleLinear()
      .domain([depthMin, depthMax])
      .range([headerHeight, headerHeight + plotHeight])
    
    // Line generator
    const createLine = (xScale) => {
      return d3.line()
        .defined(d => d.value !== null && !isNaN(d.value))
        .x(d => xScale(d.value))
        .y(d => yScale(d.depth))
    }
    
    // Draw track header box - WellCAD style
    const drawTrackHeader = (group, config, x, w) => {
      // Header background
      group.append('rect')
        .attr('x', x)
        .attr('y', 0)
        .attr('width', w)
        .attr('height', headerHeight)
        .attr('fill', '#F0F0F0')
        .attr('stroke', '#808080')
        .attr('stroke-width', 1)
      
      // Title box
      group.append('rect')
        .attr('x', x + 2)
        .attr('y', 2)
        .attr('width', w - 4)
        .attr('height', 18)
        .attr('fill', '#FFFFFF')
        .attr('stroke', '#C0C0C0')
      
      // Title text
      group.append('text')
        .attr('x', x + w / 2)
        .attr('y', 14)
        .attr('text-anchor', 'middle')
        .attr('font-size', '10px')
        .attr('font-weight', 'bold')
        .text(config.title)
      
      // Scale range box
      group.append('rect')
        .attr('x', x + 2)
        .attr('y', 22)
        .attr('width', w - 4)
        .attr('height', 14)
        .attr('fill', '#FFFFFF')
        .attr('stroke', '#C0C0C0')
      
      // Scale range text
      if (config.min !== undefined) {
        group.append('text')
          .attr('x', x + w / 2)
          .attr('y', 33)
          .attr('text-anchor', 'middle')
          .attr('font-size', '9px')
          .attr('fill', config.color || '#333')
          .text(`${config.min} ${config.unit || ''} ${config.max}`)
      }
    }
    
    // Draw track background with grid
    const drawTrackBackground = (group, x, w, startY, h) => {
      // White background
      group.append('rect')
        .attr('x', x)
        .attr('y', startY)
        .attr('width', w)
        .attr('height', h)
        .attr('fill', '#FFFFFF')
        .attr('stroke', '#808080')
      
      // Horizontal grid lines (depth)
      const depthStep = 5 // meters
      for (let d = Math.ceil(depthMin / depthStep) * depthStep; d <= depthMax; d += depthStep) {
        const y = yScale(d)
        const isMajor = d % 10 === 0
        group.append('line')
          .attr('x1', x)
          .attr('x2', x + w)
          .attr('y1', y)
          .attr('y2', y)
          .attr('stroke', isMajor ? '#C0C0C0' : '#E8E8E8')
          .attr('stroke-width', isMajor ? 1 : 0.5)
      }
    }
    
    let xOffset = 0
    
    // ========================================
    // DEPTH TRACK
    // ========================================
    const depthGroup = svg.append('g').attr('class', 'track track-depth')
    
    // Header
    depthGroup.append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', trackConfig.depth.width)
      .attr('height', headerHeight)
      .attr('fill', '#F0F0F0')
      .attr('stroke', '#808080')
    
    depthGroup.append('text')
      .attr('x', trackConfig.depth.width / 2)
      .attr('y', 14)
      .attr('text-anchor', 'middle')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .text('Depth')
    
    depthGroup.append('text')
      .attr('x', trackConfig.depth.width / 2)
      .attr('y', 30)
      .attr('text-anchor', 'middle')
      .attr('font-size', '9px')
      .text(`1m:${settings.scale / 100}m`)
    
    // Depth axis background
    depthGroup.append('rect')
      .attr('x', 0)
      .attr('y', headerHeight)
      .attr('width', trackConfig.depth.width)
      .attr('height', plotHeight)
      .attr('fill', '#FAFAFA')
      .attr('stroke', '#808080')
    
    // Depth labels
    const depthStep = 5
    for (let d = Math.ceil(depthMin / depthStep) * depthStep; d <= depthMax; d += depthStep) {
      const y = yScale(d)
      const isMajor = d % 10 === 0
      
      if (isMajor) {
        depthGroup.append('text')
          .attr('x', trackConfig.depth.width - 4)
          .attr('y', y + 3)
          .attr('text-anchor', 'end')
          .attr('font-size', '9px')
          .attr('fill', '#333')
          .text(d.toFixed(1))
      }
      
      depthGroup.append('line')
        .attr('x1', trackConfig.depth.width - (isMajor ? 8 : 4))
        .attr('x2', trackConfig.depth.width)
        .attr('y1', y)
        .attr('y2', y)
        .attr('stroke', '#666')
    }
    
    xOffset = trackConfig.depth.width
    
    // ========================================
    // GAMMA RAY TRACK  
    // ========================================
    const grKey = mapping.GR
    if (grKey && curves[grKey]) {
      const grGroup = svg.append('g')
        .attr('class', 'track track-gr')
      
      drawTrackHeader(grGroup, trackConfig.gr, xOffset, trackConfig.gr.width)
      drawTrackBackground(grGroup, xOffset, trackConfig.gr.width, headerHeight, plotHeight)
      
      // Scale
      const grScale = d3.scaleLinear()
        .domain([trackConfig.gr.min, trackConfig.gr.max])
        .range([xOffset, xOffset + trackConfig.gr.width])
      
      // Vertical grid
      grScale.ticks(5).forEach(tick => {
        svg.append('line')
          .attr('x1', grScale(tick))
          .attr('x2', grScale(tick))
          .attr('y1', headerHeight)
          .attr('y2', headerHeight + plotHeight)
          .attr('stroke', '#E0E0E0')
      })
      
      // Data
      const grData = depth.map((d, i) => ({
        depth: d,
        value: curves[grKey][i]
      }))
      
      // GR fill if enabled
      if (settings.showGrFill) {
        const area = d3.area()
          .defined(d => d.value !== null && !isNaN(d.value))
          .x0(xOffset)
          .x1(d => grScale(Math.min(d.value, trackConfig.gr.max)))
          .y(d => yScale(d.depth))
        
        grGroup.append('path')
          .datum(grData)
          .attr('d', area)
          .attr('fill', '#90EE90')
          .attr('fill-opacity', 0.4)
      }
      
      // Line
      const grLine = createLine(grScale)
      grGroup.append('path')
        .datum(grData)
        .attr('d', grLine)
        .attr('fill', 'none')
        .attr('stroke', trackConfig.gr.color)
        .attr('stroke-width', 1.5)
      
      xOffset += trackConfig.gr.width
    }
    
    // ========================================
    // RESISTIVITY TRACK
    // ========================================
    const resKey = mapping.RES_DEEP
    if (resKey && curves[resKey]) {
      const resGroup = svg.append('g')
        .attr('class', 'track track-res')
      
      // Modified header for resistivity
      const resConfig = {...trackConfig.res}
      drawTrackHeader(resGroup, resConfig, xOffset, trackConfig.res.width)
      drawTrackBackground(resGroup, xOffset, trackConfig.res.width, headerHeight, plotHeight)
      
      // Log scale
      const resScale = d3.scaleLog()
        .domain([Math.max(0.1, trackConfig.res.min), trackConfig.res.max])
        .range([xOffset, xOffset + trackConfig.res.width])
        .clamp(true)
      
      // Log grid
      ;[0.2, 1, 10, 100, 1000].forEach(tick => {
        if (tick >= trackConfig.res.min && tick <= trackConfig.res.max) {
          svg.append('line')
            .attr('x1', resScale(tick))
            .attr('x2', resScale(tick))
            .attr('y1', headerHeight)
            .attr('y2', headerHeight + plotHeight)
            .attr('stroke', tick === 1 || tick === 10 || tick === 100 ? '#C0C0C0' : '#E8E8E8')
        }
      })
      
      // Data
      const resData = depth.map((d, i) => ({
        depth: d,
        value: curves[resKey][i]
      }))
      
      // Line
      const resLine = createLine(resScale)
      resGroup.append('path')
        .datum(resData)
        .attr('d', resLine)
        .attr('fill', 'none')
        .attr('stroke', '#0000FF')
        .attr('stroke-width', 1.5)
      
      xOffset += trackConfig.res.width
    }
    
    // ========================================
    // DENSITY-NEUTRON TRACK
    // ========================================
    const densKey = mapping.DENS
    const neutKey = mapping.NEUT
    
    if ((densKey && curves[densKey]) || (neutKey && curves[neutKey])) {
      const dnGroup = svg.append('g')
        .attr('class', 'track track-dn')
      
      // Header
      dnGroup.append('rect')
        .attr('x', xOffset)
        .attr('y', 0)
        .attr('width', trackConfig.dn.width)
        .attr('height', headerHeight)
        .attr('fill', '#F0F0F0')
        .attr('stroke', '#808080')
      
      // Two sub-headers
      dnGroup.append('rect')
        .attr('x', xOffset + 2)
        .attr('y', 2)
        .attr('width', trackConfig.dn.width - 4)
        .attr('height', 16)
        .attr('fill', '#FFFFFF')
        .attr('stroke', '#C0C0C0')
      
      dnGroup.append('text')
        .attr('x', xOffset + trackConfig.dn.width / 2)
        .attr('y', 13)
        .attr('text-anchor', 'middle')
        .attr('font-size', '9px')
        .attr('font-weight', 'bold')
        .text('MC4F / MC2F')
      
      dnGroup.append('rect')
        .attr('x', xOffset + 2)
        .attr('y', 20)
        .attr('width', trackConfig.dn.width - 4)
        .attr('height', 16)
        .attr('fill', '#FFFFFF')
        .attr('stroke', '#C0C0C0')
      
      dnGroup.append('text')
        .attr('x', xOffset + trackConfig.dn.width / 2)
        .attr('y', 31)
        .attr('text-anchor', 'middle')
        .attr('font-size', '8px')
        .text('100 US/M 700')
      
      drawTrackBackground(dnGroup, xOffset, trackConfig.dn.width, headerHeight, plotHeight)
      
      // Density scale
      const densScale = d3.scaleLinear()
        .domain([trackConfig.dn.densMin, trackConfig.dn.densMax])
        .range([xOffset, xOffset + trackConfig.dn.width])
      
      // Neutron scale (reversed)
      const neutScale = d3.scaleLinear()
        .domain([trackConfig.dn.neutMax, trackConfig.dn.neutMin])
        .range([xOffset, xOffset + trackConfig.dn.width])
      
      // Density
      if (densKey && curves[densKey]) {
        const densData = depth.map((d, i) => ({
          depth: d,
          value: curves[densKey][i]
        }))
        
        const densLine = createLine(densScale)
        dnGroup.append('path')
          .datum(densData)
          .attr('d', densLine)
          .attr('fill', 'none')
          .attr('stroke', '#00AA00')
          .attr('stroke-width', 1.5)
      }
      
      // Neutron
      if (neutKey && curves[neutKey]) {
        const neutData = depth.map((d, i) => ({
          depth: d,
          value: curves[neutKey][i]
        }))
        
        const neutLine = createLine(neutScale)
        dnGroup.append('path')
          .datum(neutData)
          .attr('d', neutLine)
          .attr('fill', 'none')
          .attr('stroke', '#0000FF')
          .attr('stroke-width', 1.5)
      }
      
      xOffset += trackConfig.dn.width
    }
    
    // ========================================
    // CADE TRACK (Additional)
    // ========================================
    const cadeGroup = svg.append('g')
      .attr('class', 'track track-cade')
    
    // Just show the empty track structure
    cadeGroup.append('rect')
      .attr('x', xOffset)
      .attr('y', 0)
      .attr('width', trackConfig.cade.width)
      .attr('height', headerHeight)
      .attr('fill', '#F0F0F0')
      .attr('stroke', '#808080')
    
    cadeGroup.append('rect')
      .attr('x', xOffset + 2)
      .attr('y', 2)
      .attr('width', trackConfig.cade.width - 4)
      .attr('height', 16)
      .attr('fill', '#FFFFFF')
      .attr('stroke', '#C0C0C0')
    
    cadeGroup.append('text')
      .attr('x', xOffset + trackConfig.cade.width / 2)
      .attr('y', 13)
      .attr('text-anchor', 'middle')
      .attr('font-size', '9px')
      .attr('font-weight', 'bold')
      .text('CADE')
    
    cadeGroup.append('rect')
      .attr('x', xOffset + 2)
      .attr('y', 20)
      .attr('width', trackConfig.cade.width - 4)
      .attr('height', 16)
      .attr('fill', '#FFFFFF')
      .attr('stroke', '#C0C0C0')
    
    cadeGroup.append('text')
      .attr('x', xOffset + trackConfig.cade.width / 2)
      .attr('y', 31)
      .attr('text-anchor', 'middle')
      .attr('font-size', '8px')
      .text('90 Mil 130')
    
    drawTrackBackground(cadeGroup, xOffset, trackConfig.cade.width, headerHeight, plotHeight)
    
  }, [curveData, wellData, settings, dimensions, trackConfig])
  
  if (loading) {
    return (
      <div className="log-viewer loading">
        <div className="loading-spinner"></div>
        <span>Loading log data...</span>
      </div>
    )
  }
  
  if (!curveData) {
    return (
      <div className="log-viewer empty">
        <span>No data to display</span>
      </div>
    )
  }
  
  return (
    <div className="log-viewer" ref={containerRef}>
      <svg
        ref={svgRef}
        width={dimensions.width}
        height={dimensions.height}
        style={{ fontFamily: 'Segoe UI, Tahoma, sans-serif' }}
      />
    </div>
  )
}

export default LogViewer
