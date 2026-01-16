/**
 * GraphManagerHelpers.ts
 * Pure utility functions for graph management and UI coordination.
 * Extracted from GraphBuilderTab to reduce component complexity.
 */

import { Link, LinkMeta } from '../response_models'

// ============================================================================
// Link Creation Helpers
// ============================================================================

/**
 * Create initial link data for the backend.
 * Length is calculated from node positions and updates when nodes are moved.
 */
export const createInitialLinkData = (
  name: string,
  length: number,
  isFirstLink: boolean,
  inheritedZLevel: number = 0
) => ({
  // Required fields
  name,
  // Length is calculated from node positions - recalculated when nodes are moved
  length: Math.max(0.1, Math.min(100, length)), // Clamp to backend limits 0.1-100 units
  n_iterations: 24, // Default to match backend expectation
  has_fixed: false, // Required field
  // Optional fields
  has_constraint: false,
  is_driven: isFirstLink,
  flip: false,
  zlevel: inheritedZLevel
})

/**
 * Create a frontend Link object from backend response.
 * Uses the new structure where frontend-only fields are in `meta`.
 */
export const constructFrontendLink = (
  backendLink: any,
  startPoint: [number, number],
  endPoint: [number, number],
  defaultColor: string,
  inheritedZLevel: number = 0
): Link => {
  // Generate ID for frontend tracking
  const linkId = backendLink.id || `link_${Date.now()}_${Math.random().toString(36).substr(2, 4)}`
  
  // Create the meta object with frontend-only properties
  const meta: LinkMeta = {
    id: linkId,
    start_point: startPoint,
    end_point: endPoint,
    color: defaultColor
  }
  
  return {
    // Backend fields
    name: backendLink.name,
    length: typeof backendLink.length === 'number' ? backendLink.length : parseFloat(backendLink.length) || 1.0,
    n_iterations: backendLink.n_iterations ?? 24,
    has_fixed: backendLink.has_fixed ?? false,
    target_length: backendLink.target_length ?? null,
    target_cost_func: backendLink.target_cost_func ?? null,
    fixed_loc: backendLink.fixed_loc ?? null,
    has_constraint: backendLink.has_constraint ?? false,
    is_driven: backendLink.is_driven ?? false,
    flip: backendLink.flip ?? false,
    zlevel: backendLink.zlevel ?? inheritedZLevel,
    // Frontend meta
    meta
  }
}

// ============================================================================
// Delete & Confirmation Helpers
// ============================================================================

export interface DeleteItem {
  type: 'link' | 'node'
  id: string
  name: string
}

/**
 * Factory for creating a DeleteItem to be confirmed in dialog.
 */
export const createDeleteItem = (
  type: 'link' | 'node',
  id: string,
  name: string
): DeleteItem => ({
  type,
  id,
  name
})

// ============================================================================
// Canvas State Helpers
// ============================================================================

/**
 * Reusable canvas clear logic for button handlers.
 * Returns a function that when called will update multiple state values.
 */
export const getClearCanvasStateSetter = () => ({
  clearCurrentClick: () => null,
  clearLinkCreationMode: () => 'idle' as const,
  clearJustStartedLinkFromNode: () => false,
  clearLinkCounter: () => 0,
  clearNodeCounter: () => 0,
  clearError: () => null
})
