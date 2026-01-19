/**
 * PylinkLinks.tsx
 *
 * Data structures and utilities for Pylink links.
 *
 * ARCHITECTURE OVERVIEW:
 * ======================
 *
 * Links connect joints. The link LENGTH is implicit - it's the distance between
 * the two joints it connects. Links should NEVER change length during animation.
 *
 * Data is stored in two places:
 *
 * 1. `pylinkage.joints` - Kinematic constraints (the "truth")
 *    - Static joints: have x, y positions
 *    - Crank joints: have distance from anchor, angle
 *    - Revolute joints: have distance0 and distance1 from two parent joints
 *    - These define the GEOMETRY of the mechanism
 *
 * 2. `meta.links` - Visual/UI properties only
 *    - color: string (hex color for drawing)
 *    - connects: [jointName1, jointName2] (which joints this link spans)
 *    - isGround?: boolean (marks ground links that don't move)
 *
 * IMPORTANT: During animation, joint positions come ONLY from trajectoryData:
 *
 *   trajectoryData.trajectories[jointName][stepIndex] → [x, y]
 *
 * The frontend only needs:
 *   - currentFrame (step index)
 *   - trajectoryData.trajectories (all positions for all joints at all steps)
 *   - meta.links.connects (to know which joints to draw lines between)
 *   - meta.links.color (for rendering)
 *
 * The `meta.joints.x, y` values are ONLY used for:
 *   1. Initial rendering before simulation runs
 *   2. Editor display when dragging joints
 *
 * During animation, these are IGNORED - positions come from trajectories.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// LINK DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Minimal link metadata stored in meta.links
 *
 * Note: Link LENGTH is NOT stored! It's computed from connected joint positions.
 * During simulation, link length is enforced by the kinematic constraints
 * (distance values in Crank and Revolute joints).
 */
export interface LinkMeta {
  /** Display color (hex string, e.g., "#ff7f0e") */
  color: string

  /** Joint names this link connects [joint1, joint2] */
  connects: [string, string]

  /** Optional: marks this as a ground link (typically doesn't move) */
  isGround?: boolean
}

/**
 * Full link data for editor operations
 * Combines meta with computed properties
 */
export interface LinkWithGeometry extends LinkMeta {
  /** Link name (key in meta.links) */
  name: string

  /** Computed length based on current joint positions */
  length: number

  /** Positions of connected joints at current frame */
  p1: [number, number]
  p2: [number, number]
}

// ═══════════════════════════════════════════════════════════════════════════════
// LINK UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Calculate link length from two joint positions
 */
export function calculateLinkLength(
  p1: [number, number],
  p2: [number, number]
): number {
  const dx = p2[0] - p1[0]
  const dy = p2[1] - p1[1]
  return Math.sqrt(dx * dx + dy * dy)
}

/**
 * Get positions for all links at a given animation frame
 *
 * @param metaLinks - The meta.links object from pylinkDoc
 * @param trajectories - trajectoryData.trajectories from simulation
 * @param frameIndex - Current animation frame (step index)
 * @param fallbackPositions - Optional fallback for joints not in trajectories
 */
export function getLinksAtFrame(
  metaLinks: Record<string, LinkMeta>,
  trajectories: Record<string, [number, number][]>,
  frameIndex: number,
  fallbackPositions?: Record<string, { x: number; y: number }>
): LinkWithGeometry[] {
  const links: LinkWithGeometry[] = []

  for (const [linkName, linkMeta] of Object.entries(metaLinks)) {
    const [joint1Name, joint2Name] = linkMeta.connects

    // Get joint positions from trajectory first, then fallback
    let p1: [number, number] | undefined
    let p2: [number, number] | undefined

    // From trajectory
    if (trajectories[joint1Name]?.[frameIndex]) {
      p1 = trajectories[joint1Name][frameIndex]
    } else if (fallbackPositions?.[joint1Name]) {
      p1 = [fallbackPositions[joint1Name].x, fallbackPositions[joint1Name].y]
    }

    if (trajectories[joint2Name]?.[frameIndex]) {
      p2 = trajectories[joint2Name][frameIndex]
    } else if (fallbackPositions?.[joint2Name]) {
      p2 = [fallbackPositions[joint2Name].x, fallbackPositions[joint2Name].y]
    }

    if (p1 && p2) {
      links.push({
        name: linkName,
        color: linkMeta.color,
        connects: linkMeta.connects,
        isGround: linkMeta.isGround,
        length: calculateLinkLength(p1, p2),
        p1,
        p2
      })
    }
  }

  return links
}

/**
 * Check if a link would have inconsistent length during animation.
 *
 * This detects "stretching" links - links that connect a kinematic joint
 * (which moves) to a static joint that ISN'T part of the kinematic chain.
 * These are fundamentally invalid - rigid links can't change length.
 *
 * @param linkMeta - The link to check
 * @param jointTypes - Map of joint names to their types (from trajectoryData.jointTypes)
 * @param trajectories - Full trajectory data
 * @returns true if link would stretch (invalid), false if valid
 */
export function wouldLinkStretch(
  linkMeta: LinkMeta,
  jointTypes: Record<string, string>,
  trajectories: Record<string, [number, number][]>
): { stretches: boolean; details?: string } {
  const [j1, j2] = linkMeta.connects

  // Check if both joints have trajectories
  const j1HasTrajectory = trajectories[j1] && trajectories[j1].length > 1
  const j2HasTrajectory = trajectories[j2] && trajectories[j2].length > 1

  // If one joint moves and other doesn't (no trajectory or static single point)
  if (j1HasTrajectory !== j2HasTrajectory) {
    // Check if the positions at different frames would give different lengths
    if (j1HasTrajectory && trajectories[j2]) {
      // j1 moves, j2 is static - check if link length changes
      const staticPos = trajectories[j2][0]
      const lengths = trajectories[j1].map(p1 => calculateLinkLength(p1, staticPos))
      const minLen = Math.min(...lengths)
      const maxLen = Math.max(...lengths)

      if (maxLen - minLen > 0.01) {  // More than 0.01 units variance
        return {
          stretches: true,
          details: `Link length varies from ${minLen.toFixed(2)} to ${maxLen.toFixed(2)} across frames`
        }
      }
    } else if (j2HasTrajectory && trajectories[j1]) {
      // j2 moves, j1 is static
      const staticPos = trajectories[j1][0]
      const lengths = trajectories[j2].map(p2 => calculateLinkLength(staticPos, p2))
      const minLen = Math.min(...lengths)
      const maxLen = Math.max(...lengths)

      if (maxLen - minLen > 0.01) {
        return {
          stretches: true,
          details: `Link length varies from ${minLen.toFixed(2)} to ${maxLen.toFixed(2)} across frames`
        }
      }
    }
  }

  // Both joints have trajectories - check length at each frame
  if (j1HasTrajectory && j2HasTrajectory) {
    const lengths = trajectories[j1].map((p1, i) =>
      calculateLinkLength(p1, trajectories[j2][i])
    )
    const minLen = Math.min(...lengths)
    const maxLen = Math.max(...lengths)

    if (maxLen - minLen > 0.01) {
      return {
        stretches: true,
        details: `Link length varies from ${minLen.toFixed(2)} to ${maxLen.toFixed(2)} - mechanism over-constrained`
      }
    }
  }

  return { stretches: false }
}

/**
 * Validate all links in a mechanism
 * Returns list of problematic links
 */
export function validateLinks(
  metaLinks: Record<string, LinkMeta>,
  jointTypes: Record<string, string>,
  trajectories: Record<string, [number, number][]>
): { valid: boolean; problems: Array<{ linkName: string; issue: string }> } {
  const problems: Array<{ linkName: string; issue: string }> = []

  for (const [linkName, linkMeta] of Object.entries(metaLinks)) {
    const result = wouldLinkStretch(linkMeta, jointTypes, trajectories)
    if (result.stretches) {
      problems.push({
        linkName,
        issue: result.details || 'Link would stretch during animation'
      })
    }
  }

  return {
    valid: problems.length === 0,
    problems
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LINK CREATION
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Create minimal link metadata
 */
export function createLinkMeta(
  connects: [string, string],
  color: string,
  isGround: boolean = false
): LinkMeta {
  return {
    color,
    connects,
    ...(isGround && { isGround })
  }
}

/**
 * Default colors for links
 */
export const DEFAULT_LINK_COLORS = [
  '#1f77b4',  // blue
  '#ff7f0e',  // orange
  '#2ca02c',  // green
  '#d62728',  // red
  '#9467bd',  // purple
  '#8c564b',  // brown
  '#e377c2',  // pink
  '#7f7f7f',  // gray
  '#bcbd22',  // olive
  '#17becf'   // cyan
]

export const GROUND_LINK_COLOR = '#7f7f7f'  // gray for ground links
