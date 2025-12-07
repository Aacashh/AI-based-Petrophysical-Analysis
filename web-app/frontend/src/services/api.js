/**
 * API Service for WellLog Viewer
 * Handles all communication with Flask backend
 */

const API_BASE = '/api';

/**
 * Upload a LAS file
 */
export async function uploadLasFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || 'Upload failed');
  }
  
  return response.json();
}

/**
 * Get list of uploaded wells
 */
export async function getWells() {
  const response = await fetch(`${API_BASE}/wells`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch wells');
  }
  
  return response.json();
}

/**
 * Get well metadata
 */
export async function getWell(wellId) {
  const response = await fetch(`${API_BASE}/wells/${wellId}`);
  
  if (!response.ok) {
    throw new Error('Well not found');
  }
  
  return response.json();
}

/**
 * Get curve data for plotting
 */
export async function getWellCurves(wellId, startDepth = null, endDepth = null) {
  let url = `${API_BASE}/wells/${wellId}/curves`;
  
  const params = new URLSearchParams();
  if (startDepth !== null) params.append('start_depth', startDepth);
  if (endDepth !== null) params.append('end_depth', endDepth);
  
  if (params.toString()) {
    url += `?${params.toString()}`;
  }
  
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error('Failed to fetch curves');
  }
  
  return response.json();
}

/**
 * Delete a well
 */
export async function deleteWell(wellId) {
  const response = await fetch(`${API_BASE}/wells/${wellId}`, {
    method: 'DELETE',
  });
  
  if (!response.ok) {
    throw new Error('Failed to delete well');
  }
  
  return response.json();
}

/**
 * Export well as LAS file
 */
export async function exportLas(wellId, startDepth = null, endDepth = null) {
  let url = `${API_BASE}/wells/${wellId}/export/las`;
  
  const params = new URLSearchParams();
  if (startDepth !== null) params.append('start_depth', startDepth);
  if (endDepth !== null) params.append('end_depth', endDepth);
  
  if (params.toString()) {
    url += `?${params.toString()}`;
  }
  
  const response = await fetch(url);
  
  if (!response.ok) {
    throw new Error('Export failed');
  }
  
  // Return blob for download
  return response.blob();
}

/**
 * Health check
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return response.ok;
}
