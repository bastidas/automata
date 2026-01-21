import React, { useState, useRef, useCallback, useEffect } from 'react'
import {
  Box,
  Typography,
  Paper,
  IconButton,
  Tooltip,
  Divider,
  Chip,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  FormControlLabel,
  Switch,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions
} from '@mui/material'
import {
  FooterToolbar,
  TOOLS,
  ToolMode,
  StatusMessage,
  StatusType,
  LinkCreationState,
  initialLinkCreationState,
  DragState,
  initialDragState,
  GroupSelectionState,
  initialGroupSelectionState,
  PolygonDrawState,
  initialPolygonDrawState,
  MeasureState,
  initialMeasureState,
  MeasurementMarker,
  DrawnObjectsState,
  initialDrawnObjectsState,
  createDrawnObject,
  MoveGroupState,
  initialMoveGroupState,
  MergePolygonState,
  initialMergePolygonState,
  PathDrawState,
  initialPathDrawState,
  TargetPath,
  createTargetPath,
  findNearestJoint,
  findNearestLink,
  findMergeTarget,
  calculateMergeResult,
  generateJointName,
  generateLinkName,
  calculateDistance,
  getDefaultColor,
  calculateLinkDeletionResult,
  calculateJointDeletionResult,
  findConnectedMechanism,
  findElementsInBox,
  isPointInPolygon,
  areLinkEndpointsInPolygon,
  MERGE_THRESHOLD,
  DraggableToolbar,
  ToolbarToggleButtons,
  TOOLBAR_CONFIGS,
  ToolbarPosition,
  JointEditModal,
  JointData,
  LinkEditModal,
  LinkData
} from './PylinkBuilderTools'
import {
  jointColors,
  getCyclicColor,
  ColorCycleType
} from '../theme'
import {
  useAnimation,
  useSimulation,
  canSimulate
} from './PylinkAnimateSimulate'
import {
  validateLinks,
  LinkMeta as PylinkLinkMeta
} from './PylinkLinks'

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION CONSTANTS (defaults - actual values are in state for Settings panel)
// ═══════════════════════════════════════════════════════════════════════════════

/** Range for simulation steps input */
const MIN_SIMULATION_STEPS = 4
const MAX_SIMULATION_STEPS = 256

/** Default values for settings (used to initialize state) */
const DEFAULT_AUTO_SIMULATE_DELAY_MS = 5
const DEFAULT_JOINT_MERGE_RADIUS = MERGE_THRESHOLD
const DEFAULT_SIMULATION_STEPS = 64
const DEFAULT_TRAJECTORY_COLOR_CYCLE: ColorCycleType = 'rainbow'
// ═══════════════════════════════════════════════════════════════════════════════
// PYLINK DATA TYPES - Native pylinkage format
// ═══════════════════════════════════════════════════════════════════════════════

interface JointRef {
  ref: string
}

interface StaticJoint {
  type: 'Static'
  name: string
  x: number
  y: number
}

interface CrankJoint {
  type: 'Crank'
  name: string
  joint0: JointRef
  distance: number
  angle: number
}

interface RevoluteJoint {
  type: 'Revolute'
  name: string
  joint0: JointRef
  joint1: JointRef
  distance0: number
  distance1: number
}

type PylinkJoint = StaticJoint | CrankJoint | RevoluteJoint

interface PylinkageData {
  name: string
  joints: PylinkJoint[]
  solve_order: string[]
}

// UI Metadata types
interface JointMeta {
  color: string
  zlevel: number
  x?: number  // UI position (single source of truth for non-Static joints)
  y?: number
  show_path?: boolean  // Whether to show trajectory path for this joint (Crank/Revolute only)
}

interface LinkMeta {
  color: string
  connects: string[]
  isGround?: boolean  // True if this is a ground/anchored link (connects static joints)
}

interface UIMeta {
  joints: Record<string, JointMeta>
  links: Record<string, LinkMeta>
}

interface PylinkDocument {
  name: string
  pylinkage: PylinkageData
  meta: UIMeta
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

const PylinkBuilderTab: React.FC = () => {
  // Canvas scaling: 6 pixels = 1 unit
  const PIXELS_PER_UNIT = 6
  const CANVAS_MIN_WIDTH_PX = 800
  const CANVAS_MIN_HEIGHT_PX = 500

  const pixelsToUnits = (pixels: number) => pixels / PIXELS_PER_UNIT
  const unitsToPixels = (units: number) => units * PIXELS_PER_UNIT

  // State
  const [toolMode, setToolMode] = useState<ToolMode>('select')
  const [openToolbars, setOpenToolbars] = useState<Set<string>>(new Set(['tools', 'more']))
  const [toolbarPositions, setToolbarPositions] = useState<Record<string, ToolbarPosition>>({})
  const [hoveredTool, setHoveredTool] = useState<ToolMode | null>(null)

  // Edit modal state - holds data for the modal dialogs
  const [editingJointData, setEditingJointData] = useState<JointData | null>(null)
  const [editingLinkData, setEditingLinkData] = useState<LinkData | null>(null)

  // Document state - using pylink native format
  const [pylinkDoc, setPylinkDoc] = useState<PylinkDocument>({
    name: 'untitled',
    pylinkage: {
      name: 'untitled',
      joints: [],
      solve_order: []
    },
    meta: {
      joints: {},
      links: {}
    }
  })

  // UI state
  const [selectedJoints, setSelectedJoints] = useState<string[]>([])
  const [selectedLinks, setSelectedLinks] = useState<string[]>([])
  const [hoveredJoint, setHoveredJoint] = useState<string | null>(null)
  const [hoveredLink, setHoveredLink] = useState<string | null>(null)

  // Status message system
  const [statusMessage, setStatusMessage] = useState<StatusMessage | null>(null)

  // Link creation state
  const [linkCreationState, setLinkCreationState] = useState<LinkCreationState>(initialLinkCreationState)
  const [previewLine, setPreviewLine] = useState<{ start: [number, number]; end: [number, number] } | null>(null)

  // Drag state for move/merge functionality
  const [dragState, setDragState] = useState<DragState>(initialDragState)

  // Group selection state (for box selection)
  const [groupSelectionState, setGroupSelectionState] = useState<GroupSelectionState>(initialGroupSelectionState)

  // Polygon drawing state
  const [polygonDrawState, setPolygonDrawState] = useState<PolygonDrawState>(initialPolygonDrawState)

  // Measure tool state
  const [measureState, setMeasureState] = useState<MeasureState>(initialMeasureState)
  const [measurementMarkers, setMeasurementMarkers] = useState<MeasurementMarker[]>([])

  // Merge polygon state (for merging polygons with links)
  const [mergePolygonState, setMergePolygonState] = useState<MergePolygonState>(initialMergePolygonState)

  // Target path drawing state (for trajectory optimization)
  const [pathDrawState, setPathDrawState] = useState<PathDrawState>(initialPathDrawState)
  const [targetPaths, setTargetPaths] = useState<TargetPath[]>([])
  const [selectedPathId, setSelectedPathId] = useState<string | null>(null)

  // Optimization state
  const [isOptimizing, setIsOptimizing] = useState(false)
  const [preOptimizationDoc, setPreOptimizationDoc] = useState<PylinkDocument | null>(null)
  const [optimizationResult, setOptimizationResult] = useState<{
    success: boolean
    initialError: number
    finalError: number
    message: string
    iterations?: number
    executionTimeMs?: number
    optimizedDimensions?: Record<string, number>
    originalDimensions?: Record<string, number>
  } | null>(null)

  // Optimization hyperparameters
  const [optMethod, setOptMethod] = useState<'pso' | 'pylinkage' | 'scipy' | 'powell' | 'nelder-mead'>('pylinkage')
  const [optNParticles, setOptNParticles] = useState(32)
  const [optIterations, setOptIterations] = useState(512)
  const [optMaxIterations, setOptMaxIterations] = useState(100)
  const [optTolerance, setOptTolerance] = useState(1e-6)
  const [optBoundsFactor, setOptBoundsFactor] = useState(2.0)
  const [optMinLength, setOptMinLength] = useState(5)
  const [optVerbose, setOptVerbose] = useState(true)

  // Trajectory preprocessing state
  const [prepEnableSmooth, setPrepEnableSmooth] = useState(true)
  const [prepSmoothWindow, setPrepSmoothWindow] = useState(4)
  const [prepSmoothPolyorder, setPrepSmoothPolyorder] = useState(3)
  const [prepSmoothMethod, setPrepSmoothMethod] = useState<'savgol' | 'moving_avg' | 'gaussian'>('savgol')
  const [prepEnableResample, setPrepEnableResample] = useState(true)
  const [prepTargetNSteps, setPrepTargetNSteps] = useState(DEFAULT_SIMULATION_STEPS)
  const [prepResampleMethod, setPrepResampleMethod] = useState<'parametric' | 'cubic' | 'linear'>('parametric')
  const [isPreprocessing, setIsPreprocessing] = useState(false)
  const [preprocessResult, setPreprocessResult] = useState<{
    originalPoints: number
    outputPoints: number
    analysis: Record<string, unknown>
  } | null>(null)

  // Drawn objects state (polygons, shapes that can be attached to links)
  const [drawnObjects, setDrawnObjects] = useState<DrawnObjectsState>(initialDrawnObjectsState)

  // Delete confirmation dialog state
  const [deleteConfirmDialog, setDeleteConfirmDialog] = useState<{
    open: boolean
    joints: string[]
    links: string[]
  }>({ open: false, joints: [], links: [] })

  // Move group state (for moving groups of joints and/or DrawnObjects)
  const [moveGroupState, setMoveGroupState] = useState<MoveGroupState>(initialMoveGroupState)

  // Simulation state - using hooks from PylinkAnimateSimulate
  const [simulationSteps, setSimulationSteps] = useState(DEFAULT_SIMULATION_STEPS)
  const [simulationStepsInput, setSimulationStepsInput] = useState(String(DEFAULT_SIMULATION_STEPS))  // Local input for debouncing
  const [mechanismVersion, setMechanismVersion] = useState(0)  // Increments on any change
  const [showTrajectory, setShowTrajectory] = useState(true)  // Toggle trajectory visibility

  // Settings state (configurable via Settings panel)
  const [autoSimulateDelayMs, setAutoSimulateDelayMs] = useState(DEFAULT_AUTO_SIMULATE_DELAY_MS)
  const [jointMergeRadius, setJointMergeRadius] = useState(DEFAULT_JOINT_MERGE_RADIUS)
  const [trajectoryColorCycle, setTrajectoryColorCycle] = useState<ColorCycleType>(DEFAULT_TRAJECTORY_COLOR_CYCLE)
  const [darkMode, setDarkMode] = useState(false)
  const [showGrid, setShowGrid] = useState(true)
  const [showJointLabels, setShowJointLabels] = useState(false)
  const [showLinkLabels, setShowLinkLabels] = useState(false)

  // Canvas/Visualization settings
  const [canvasBgColor, setCanvasBgColor] = useState<'default' | 'white' | 'cream' | 'dark'>('default')
  const [jointSize, setJointSize] = useState(8)  // 3-16px
  const [linkThickness, setLinkThickness] = useState(8)  // 1-16px
  const [trajectoryDotSize, setTrajectoryDotSize] = useState(4)  // 2-6px
  const [selectionHighlightColor, setSelectionHighlightColor] = useState<'blue' | 'orange' | 'green' | 'purple'>('blue')
  const [showMeasurementUnits, setShowMeasurementUnits] = useState(true)
  const [trajectoryStyle, setTrajectoryStyle] = useState<'dots' | 'line' | 'both'>('both')

  // Derived selection color
  const selectionColorMap = { blue: '#1976d2', orange: '#FA8112', green: '#2e7d32', purple: '#9c27b0' }
  const selectionColor = selectionColorMap[selectionHighlightColor]

  // ═══════════════════════════════════════════════════════════════════════════════
  // HIGHLIGHT HELPER - Creates consistent glow/outline effects for selected objects
  // ═══════════════════════════════════════════════════════════════════════════════
  type HighlightType = 'none' | 'selected' | 'hovered' | 'move_group' | 'merge'
  type ObjectType = 'joint' | 'link' | 'polygon'

  /**
   * Get glow filter ID based on color - maps colors to pre-defined SVG filters
   * Objects glow in their own original color, not the selection color
   */
  const getGlowFilterForColor = useCallback((color: string): string => {
    const colorLower = color.toLowerCase()

    // Joint type colors
    if (colorLower === '#e74c3c' || colorLower === jointColors.static.toLowerCase()) return 'url(#glow-static)'
    if (colorLower === '#f39c12' || colorLower === jointColors.crank.toLowerCase()) return 'url(#glow-crank)'
    if (colorLower === '#2196f3' || colorLower === jointColors.pivot.toLowerCase()) return 'url(#glow-pivot)'

    // Graph colors (D3 palette)
    if (colorLower === '#1f77b4') return 'url(#glow-blue)'
    if (colorLower === '#ff7f0e') return 'url(#glow-orange)'
    if (colorLower === '#2ca02c') return 'url(#glow-green)'
    if (colorLower === '#d62728') return 'url(#glow-red)'
    if (colorLower === '#9467bd') return 'url(#glow-purple)'
    if (colorLower === '#8c564b') return 'url(#glow-brown)'
    if (colorLower === '#e377c2') return 'url(#glow-pink)'
    if (colorLower === '#7f7f7f') return 'url(#glow-gray)'
    if (colorLower === '#bcbd22') return 'url(#glow-olive)'
    if (colorLower === '#17becf') return 'url(#glow-cyan)'

    // Default to blue glow for unknown colors
    return 'url(#glow-blue)'
  }, [])

  /**
   * Get highlight styling for objects (joints, links, polygons)
   * Objects glow in their ORIGINAL color when selected/hovered
   * Move group uses grey glow
   */
  const getHighlightStyle = useCallback((
    objectType: ObjectType,
    highlightType: HighlightType,
    baseColor: string,  // The object's original color (for glow)
    baseStrokeWidth: number
  ): { stroke: string; strokeWidth: number; filter?: string } => {
    // No highlight - return base styling
    if (highlightType === 'none') {
      return { stroke: baseColor, strokeWidth: baseStrokeWidth }
    }

    // Different stroke widths based on object type and highlight state
    const glowStrokeWidth = objectType === 'joint'
      ? baseStrokeWidth + 1
      : objectType === 'link'
        ? baseStrokeWidth + 2
        : baseStrokeWidth + 1

    // Move group and merge use special colors, selected/hovered use object's original color
    switch (highlightType) {
      case 'move_group':
        return {
          stroke: jointColors.moveGroup,
          strokeWidth: glowStrokeWidth,
          filter: 'url(#glow-movegroup)'
        }
      case 'merge':
        return {
          stroke: jointColors.mergeHighlight,
          strokeWidth: glowStrokeWidth,
          filter: 'url(#glow-merge)'
        }
      case 'selected':
      case 'hovered':
      default:
        // Glow in the object's original color
        return {
          stroke: baseColor,
          strokeWidth: glowStrokeWidth,
          filter: getGlowFilterForColor(baseColor)
        }
    }
  }, [getGlowFilterForColor])

  // Canvas dimensions
  const [canvasDimensions, setCanvasDimensions] = useState({ width: 1200, height: 700 })

  const canvasRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Animation: positions override during playback
  const [animatedPositions, setAnimatedPositions] = useState<Record<string, [number, number]> | null>(null)

  // Update canvas dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect()
        setCanvasDimensions({
          width: rect.width,
          height: Math.max(rect.height - 48, CANVAS_MIN_HEIGHT_PX) // Account for footer
        })
      }
    }

    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  // Apply dark mode class to document body
  useEffect(() => {
    if (darkMode) {
      document.body.classList.add('dark-mode')
    } else {
      document.body.classList.remove('dark-mode')
    }
  }, [darkMode])

  // Debounced simulation steps update - wait 500ms after user stops typing
  const simulationStepsInputRef = useRef(simulationStepsInput)
  simulationStepsInputRef.current = simulationStepsInput
  const prevSimulationStepsRef = useRef(simulationSteps)

  useEffect(() => {
    const timer = setTimeout(() => {
      const val = parseInt(simulationStepsInputRef.current)
      if (!isNaN(val)) {
        const clamped = Math.max(MIN_SIMULATION_STEPS, Math.min(MAX_SIMULATION_STEPS, val))
        setSimulationSteps(prev => {
          if (clamped !== prev) {
            // Update input to show clamped value
            setSimulationStepsInput(String(clamped))
            return clamped
          }
          return prev
        })
      }
    }, 500)
    return () => clearTimeout(timer)
  }, [simulationStepsInput])

  // Trigger simulation when simulationSteps changes (after debounce)
  useEffect(() => {
    if (simulationSteps !== prevSimulationStepsRef.current) {
      prevSimulationStepsRef.current = simulationSteps
      // Trigger mechanism change to re-run simulation with new step count
      setMechanismVersion(v => v + 1)
    }
  }, [simulationSteps])

  // Helper to show status messages
  const showStatus = useCallback((text: string, type: StatusType = 'info', duration?: number) => {
    setStatusMessage({ text, type, timestamp: Date.now() })

    if (duration && type !== 'action') {
      setTimeout(() => {
        setStatusMessage(prev => {
          if (prev && prev.text === text) return null
          return prev
        })
      }, duration)
    }
  }, [])

  // Clear status
  const clearStatus = useCallback(() => {
    setStatusMessage(null)
  }, [])

  // Trigger mechanism change (for auto-simulation)
  const triggerMechanismChange = useCallback(() => {
    setMechanismVersion(v => v + 1)
  }, [])

  // Handle animation frame changes - update joint positions visually
  const handleAnimationFrameChange = useCallback((_frame: number) => {
    // This will be called by the animation hook when the frame changes
    // We update animatedPositions in the useEffect below
  }, [])

  // Simulation hook - handles trajectory computation and auto-simulate
  const {
    isSimulating,
    trajectoryData,
    runSimulation,
    clearTrajectory,
    setAutoSimulateEnabled,
    autoSimulateEnabled
  } = useSimulation({
    pylinkDoc,
    simulationSteps,
    autoSimulateDelayMs,
    autoSimulateEnabled: true,  // Start with continuous simulation ON by default
    mechanismVersion,
    showStatus
  })

  // Animation hook - handles playback of simulation frames
  const {
    animationState,
    play: playAnimation,
    pause: pauseAnimation,
    stop: stopAnimation,
    reset: _resetAnimation,
    setFrame: _setAnimationFrame,
    setPlaybackSpeed,
    getAnimatedPositions
  } = useAnimation({
    trajectoryData,
    onFrameChange: handleAnimationFrameChange,
    frameIntervalMs: 50  // 20fps default
  })

  // Update animated positions when animation frame changes
  useEffect(() => {
    if (animationState.isAnimating) {
      const positions = getAnimatedPositions()
      setAnimatedPositions(positions)
    } else if (!animationState.isAnimating && animationState.currentFrame === 0) {
      // Reset to original positions when stopped
      setAnimatedPositions(null)
    }
  }, [animationState.isAnimating, animationState.currentFrame, getAnimatedPositions])

  // Track invalid links that would stretch during animation
  const [stretchingLinks, setStretchingLinks] = useState<string[]>([])

  // Validate links after simulation completes - detect links that would stretch
  useEffect(() => {
    if (trajectoryData && trajectoryData.trajectories) {
      // Convert local LinkMeta format to PylinkLinkMeta format for validation
      const linksForValidation: Record<string, PylinkLinkMeta> = {}
      for (const [name, link] of Object.entries(pylinkDoc.meta.links)) {
        if (link.connects && link.connects.length === 2) {
          linksForValidation[name] = {
            color: link.color,
            connects: [link.connects[0], link.connects[1]] as [string, string],
            isGround: link.isGround
          }
        }
      }

      const validation = validateLinks(
        linksForValidation,
        trajectoryData.jointTypes,
        trajectoryData.trajectories
      )

      if (!validation.valid) {
        // Store problematic links
        const problemLinkNames = validation.problems.map(p => p.linkName)
        setStretchingLinks(problemLinkNames)

        // Show warning for each problematic link
        const problemLinksStr = problemLinkNames.join(', ')
        showStatus(
          `⚠️ Invalid mechanism: ${problemLinksStr} would stretch during animation. ` +
          `These links connect moving joints to fixed joints without proper constraints.`,
          'warning',
          5000
        )
        console.warn('Link validation problems:', validation.problems)

        // Stop any running animation - can't animate invalid mechanism
        if (animationState.isAnimating) {
          pauseAnimation()
        }
      } else {
        setStretchingLinks([])
      }
    } else {
      setStretchingLinks([])
    }
  }, [trajectoryData, pylinkDoc.meta.links, showStatus, animationState.isAnimating, pauseAnimation])

  // Cancel current action
  const cancelAction = useCallback(() => {
    if (linkCreationState.isDrawing) {
      setLinkCreationState(initialLinkCreationState)
      setPreviewLine(null)
      showStatus('Link creation cancelled', 'info', 2000)
    }
    if (dragState.isDragging) {
      setDragState(initialDragState)
      showStatus('Drag cancelled', 'info', 2000)
    }
    if (groupSelectionState.isSelecting) {
      setGroupSelectionState(initialGroupSelectionState)
      showStatus('Group selection cancelled', 'info', 2000)
    }
    if (polygonDrawState.isDrawing) {
      setPolygonDrawState(initialPolygonDrawState)
      showStatus('Polygon drawing cancelled', 'info', 2000)
    }
    if (measureState.isMeasuring) {
      setMeasureState(initialMeasureState)
      showStatus('Measurement cancelled', 'info', 2000)
    }
    if (mergePolygonState.step !== 'idle' && mergePolygonState.step !== 'awaiting_selection') {
      setMergePolygonState(initialMergePolygonState)
      setDrawnObjects(prev => ({ ...prev, selectedIds: [] }))
      setSelectedLinks([])
      showStatus('Merge cancelled', 'info', 2000)
    }
    if (pathDrawState.isDrawing) {
      setPathDrawState(initialPathDrawState)
      showStatus('Path drawing cancelled', 'info', 2000)
    }
    setToolMode('select')
  }, [linkCreationState.isDrawing, dragState.isDragging, groupSelectionState.isSelecting, polygonDrawState.isDrawing, measureState.isMeasuring, mergePolygonState.step, pathDrawState.isDrawing, showStatus])

  // Get joints connected to a given joint via visual links
  // This is used to determine the correct parent joints for Revolute type conversion
  const getConnectedJointsFromLinks = useCallback((
    jointName: string,
    links: Record<string, { color: string; connects: string[] }>
  ): string[] => {
    const connected: string[] = []
    Object.values(links).forEach(link => {
      if (link.connects.includes(jointName)) {
        // Add the OTHER joint in the link (not the one we're looking for)
        link.connects.forEach(connectedJoint => {
          if (connectedJoint !== jointName && !connected.includes(connectedJoint)) {
            connected.push(connectedJoint)
          }
        })
      }
    })
    return connected
  }, [])

  // Get visual position for any joint
  // SINGLE SOURCE OF TRUTH:
  // - Static joints: position from pylinkage data (x, y)
  // - Crank/Revolute joints: position from meta.joints (x, y) if available,
  //   otherwise calculate from parent relationships
  const getJointPosition = useCallback((jointName: string): [number, number] | null => {
    // When animating, use animated positions for moving joints
    if (animatedPositions && animatedPositions[jointName]) {
      return animatedPositions[jointName]
    }

    const joint = pylinkDoc.pylinkage.joints.find(j => j.name === jointName)
    if (!joint) return null

    // For Static joints, always use the stored x, y
    if (joint.type === 'Static') {
      return [joint.x, joint.y]
    }

    // For non-Static joints, check meta.joints first (single source of truth for UI position)
    const meta = pylinkDoc.meta.joints[jointName]
    if (meta?.x !== undefined && meta?.y !== undefined) {
      return [meta.x, meta.y]
    }

    // Fallback: calculate position from parent relationships (for initial load/demo)
    if (joint.type === 'Crank') {
      const parent = pylinkDoc.pylinkage.joints.find(j => j.name === joint.joint0.ref)
      if (parent && parent.type === 'Static') {
        const x = parent.x + joint.distance * Math.cos(joint.angle)
        const y = parent.y + joint.distance * Math.sin(joint.angle)
        return [x, y]
      }
    } else if (joint.type === 'Revolute') {
      // For Revolute, use circle-circle intersection or approximate from distances
      const parent0 = pylinkDoc.pylinkage.joints.find(j => j.name === joint.joint0.ref)
      const parent1 = pylinkDoc.pylinkage.joints.find(j => j.name === joint.joint1.ref)
      if (parent0 && parent1) {
        const pos0 = getJointPosition(parent0.name)
        const pos1 = getJointPosition(parent1.name)
        if (pos0 && pos1) {
          // Circle-circle intersection to find the joint position
          const d0 = joint.distance0
          const d1 = joint.distance1
          const dx = pos1[0] - pos0[0]
          const dy = pos1[1] - pos0[1]
          const d = Math.sqrt(dx * dx + dy * dy)

          if (d > 0 && d <= d0 + d1 && d >= Math.abs(d0 - d1)) {
            // Valid triangle, compute intersection point
            const a = (d0 * d0 - d1 * d1 + d * d) / (2 * d)
            const h = Math.sqrt(Math.max(0, d0 * d0 - a * a))
            const px = pos0[0] + (a * dx) / d
            const py = pos0[1] + (a * dy) / d
            // Return one of the two intersection points (above the line)
            const x = px - (h * dy) / d
            const y = py + (h * dx) / d
            return [x, y]
          }
          // Fallback if geometry doesn't work
          return [(pos0[0] + pos1[0]) / 2, (pos0[1] + pos1[1]) / 2 - 20]
        }
      }
    }
    return null
  }, [pylinkDoc.pylinkage.joints, pylinkDoc.meta.joints, animatedPositions])

  // Get all joints with their positions for snapping
  const getJointsWithPositions = useCallback(() => {
    return pylinkDoc.pylinkage.joints.map(joint => ({
      name: joint.name,
      position: getJointPosition(joint.name)
    }))
  }, [pylinkDoc.pylinkage.joints, getJointPosition])

  // ═══════════════════════════════════════════════════════════════════════════════
  // MODAL DATA BUILDERS - Create data for edit modals
  // ═══════════════════════════════════════════════════════════════════════════════

  /**
   * Build JointData for the joint edit modal
   */
  const buildJointData = useCallback((jointName: string): JointData | null => {
    const joint = pylinkDoc.pylinkage.joints.find(j => j.name === jointName)
    if (!joint) return null

    const position = getJointPosition(jointName)
    const meta = pylinkDoc.meta.joints[jointName]

    // Find connected links
    const connectedLinks = Object.entries(pylinkDoc.meta.links)
      .filter(([_, linkMeta]) => linkMeta.connects.includes(jointName))
      .map(([linkName]) => linkName)

    const baseData: JointData = {
      name: joint.name,
      type: joint.type,
      position,
      connectedLinks,
      showPath: meta?.show_path ?? false
    }

    // Add type-specific data
    if (joint.type === 'Crank') {
      return {
        ...baseData,
        parentJoint: joint.joint0.ref,
        distance: joint.distance,
        angle: joint.angle
      }
    } else if (joint.type === 'Revolute') {
      return {
        ...baseData,
        parentJoint: joint.joint0.ref,
        parentJoint2: joint.joint1.ref,
        distance: joint.distance0,
        distance2: joint.distance1
      }
    }

    return baseData
  }, [pylinkDoc.pylinkage.joints, pylinkDoc.meta.joints, pylinkDoc.meta.links, getJointPosition])

  /**
   * Build LinkData for the link edit modal
   */
  const buildLinkData = useCallback((linkName: string): LinkData | null => {
    const linkMeta = pylinkDoc.meta.links[linkName]
    if (!linkMeta || linkMeta.connects.length < 2) return null

    const pos0 = getJointPosition(linkMeta.connects[0])
    const pos1 = getJointPosition(linkMeta.connects[1])
    const length = pos0 && pos1 ? calculateDistance(pos0, pos1) : null

    // Get link index for default color
    const linkIndex = Object.keys(pylinkDoc.meta.links).indexOf(linkName)

    return {
      name: linkName,
      color: linkMeta.color || getDefaultColor(linkIndex),
      connects: [linkMeta.connects[0], linkMeta.connects[1]],
      length,
      isGround: linkMeta.isGround || false,
      jointPositions: [pos0, pos1]
    }
  }, [pylinkDoc.meta.links, getJointPosition])

  /**
   * Open joint edit modal
   */
  const openJointEditModal = useCallback((jointName: string) => {
    const data = buildJointData(jointName)
    if (data) {
      setEditingJointData(data)
    }
  }, [buildJointData])

  /**
   * Open link edit modal
   */
  const openLinkEditModal = useCallback((linkName: string) => {
    const data = buildLinkData(linkName)
    if (data) {
      setEditingLinkData(data)
    }
  }, [buildLinkData])

  // Get all links with their positions for snapping
  const getLinksWithPositions = useCallback(() => {
    return Object.entries(pylinkDoc.meta.links).map(([name, meta]) => {
      const start = getJointPosition(meta.connects[0])
      const end = getJointPosition(meta.connects[1])
      return { name, start, end }
    })
  }, [pylinkDoc.meta.links, getJointPosition])

  // Enter move group mode - allows moving selected joints and/or DrawnObjects as a rigid body
  const enterMoveGroupMode = useCallback((
    jointNames: string[],
    drawnObjectIds: string[] = []
  ) => {
    // Store original positions of all joints
    const startPositions: Record<string, [number, number]> = {}
    jointNames.forEach(jointName => {
      const pos = getJointPosition(jointName)
      if (pos) {
        startPositions[jointName] = pos
      }
    })

    // Store original positions of all drawn object points
    const drawnObjectStartPositions: Record<string, [number, number][]> = {}
    drawnObjectIds.forEach(id => {
      const obj = drawnObjects.objects.find(o => o.id === id)
      if (obj) {
        drawnObjectStartPositions[id] = [...obj.points]
      }
    })

    setMoveGroupState({
      isActive: true,
      isDragging: false,
      joints: jointNames,
      drawnObjectIds,
      startPositions,
      drawnObjectStartPositions,
      dragStartPoint: null
    })

    const totalItems = jointNames.length + drawnObjectIds.length
    showStatus(`Move mode: ${totalItems} items selected — click and drag to move`, 'action')
  }, [getJointPosition, drawnObjects.objects, showStatus])

  // Exit move group mode
  const exitMoveGroupMode = useCallback(() => {
    setMoveGroupState(initialMoveGroupState)
    setToolMode('select')
    clearStatus()
  }, [clearStatus])

  // Delete a link and any orphan joints
  const deleteLink = useCallback((linkName: string) => {
    const result = calculateLinkDeletionResult(linkName, pylinkDoc.meta.links)

    // Create new state
    const newLinks = { ...pylinkDoc.meta.links }
    const newJoints = { ...pylinkDoc.meta.joints }
    const newPylinkageJoints = pylinkDoc.pylinkage.joints.filter(
      j => !result.jointsToDelete.includes(j.name)
    )
    const newSolveOrder = pylinkDoc.pylinkage.solve_order.filter(
      name => !result.jointsToDelete.includes(name)
    )

    // Remove the link
    delete newLinks[linkName]

    // Remove orphan joint metadata
    result.jointsToDelete.forEach(jointName => {
      delete newJoints[jointName]
    })

    // Clear trajectory and trigger auto-simulation
    clearTrajectory()
    triggerMechanismChange()

    setPylinkDoc({
      ...pylinkDoc,
      pylinkage: {
        ...pylinkDoc.pylinkage,
        joints: newPylinkageJoints,
        solve_order: newSolveOrder
      },
      meta: {
        joints: newJoints,
        links: newLinks
      }
    })

    // Clear selections
    setSelectedLinks([])
    setSelectedJoints([])

    const orphanMsg = result.jointsToDelete.length > 0
      ? ` + ${result.jointsToDelete.length} orphan(s)`
      : ''
    showStatus(`Deleted ${linkName}${orphanMsg}`, 'success', 2500)
  }, [pylinkDoc, showStatus])

  // Delete a joint and all connected links (one degree out), plus any resulting orphans
  const deleteJoint = useCallback((jointName: string) => {
    const result = calculateJointDeletionResult(jointName, pylinkDoc.meta.links)

    // Create new state - filter out all joints to be deleted (including orphans)
    const newLinks = { ...pylinkDoc.meta.links }
    const newJoints = { ...pylinkDoc.meta.joints }
    const newPylinkageJoints = pylinkDoc.pylinkage.joints.filter(
      j => !result.jointsToDelete.includes(j.name)
    )
    const newSolveOrder = pylinkDoc.pylinkage.solve_order.filter(
      name => !result.jointsToDelete.includes(name)
    )

    // Remove all connected links
    result.linksToDelete.forEach(linkName => {
      delete newLinks[linkName]
    })

    // Remove all joint metadata (including orphans)
    result.jointsToDelete.forEach(jName => {
      delete newJoints[jName]
    })

    // Clear trajectory and trigger auto-simulation
    clearTrajectory()
    triggerMechanismChange()

    setPylinkDoc({
      ...pylinkDoc,
      pylinkage: {
        ...pylinkDoc.pylinkage,
        joints: newPylinkageJoints,
        solve_order: newSolveOrder
      },
      meta: {
        joints: newJoints,
        links: newLinks
      }
    })

    // Clear selections
    setSelectedLinks([])
    setSelectedJoints([])

    // Build status message (result.jointsToDelete includes the original joint + orphans)
    const orphanCount = result.jointsToDelete.length - 1
    const parts: string[] = [`Deleted ${jointName}`]
    if (result.linksToDelete.length > 0) {
      parts.push(`${result.linksToDelete.length} link(s)`)
    }
    if (orphanCount > 0) {
      parts.push(`${orphanCount} orphan(s)`)
    }
    showStatus(parts.join(' + '), 'success', 2500)
  }, [pylinkDoc, showStatus])

  // Move a joint to a new position (converts non-Static joints to Static)
  const moveJoint = useCallback((jointName: string, newPosition: [number, number]) => {
    const jointIndex = pylinkDoc.pylinkage.joints.findIndex(j => j.name === jointName)
    if (jointIndex === -1) return

    const currentJoint = pylinkDoc.pylinkage.joints[jointIndex]
    const newJoints = [...pylinkDoc.pylinkage.joints]
    const newMetaJoints = { ...pylinkDoc.meta.joints }

    if (currentJoint.type === 'Static') {
      // Static joints: update position directly in pylinkage data
      newJoints[jointIndex] = {
        type: 'Static',
        name: jointName,
        x: newPosition[0],
        y: newPosition[1]
      } as StaticJoint
      // Clear meta position since Static uses pylinkage x,y
      if (newMetaJoints[jointName]) {
        newMetaJoints[jointName] = { ...newMetaJoints[jointName], x: undefined, y: undefined }
      }
    } else if (currentJoint.type === 'Crank') {
      // Crank joints: preserve type and rotation speed, update distance to match new position
      const parentName = currentJoint.joint0.ref
      const parentPos = getJointPosition(parentName)
      if (parentPos) {
        const distance = calculateDistance(parentPos, newPosition)
        // Preserve the original rotation speed (angle per step)
        // The initial angle is determined by meta.joints x,y position
        newJoints[jointIndex] = {
          type: 'Crank',
          name: jointName,
          joint0: currentJoint.joint0,
          distance: distance,
          angle: currentJoint.angle  // Keep original rotation speed
        } as CrankJoint
        // Update meta position for UI rendering (this determines initial position)
        newMetaJoints[jointName] = {
          ...newMetaJoints[jointName] || { color: '', zlevel: 0 },
          x: newPosition[0],
          y: newPosition[1]
        }
      }
    } else if (currentJoint.type === 'Revolute') {
      // Revolute joints: preserve type, update distances to match new position
      const parent0Name = currentJoint.joint0.ref
      const parent1Name = currentJoint.joint1.ref
      const parent0Pos = getJointPosition(parent0Name)
      const parent1Pos = getJointPosition(parent1Name)
      if (parent0Pos && parent1Pos) {
        const distance0 = calculateDistance(parent0Pos, newPosition)
        const distance1 = calculateDistance(parent1Pos, newPosition)
        newJoints[jointIndex] = {
          type: 'Revolute',
          name: jointName,
          joint0: currentJoint.joint0,
          joint1: currentJoint.joint1,
          distance0: distance0,
          distance1: distance1
        } as RevoluteJoint
        // Update meta position for UI rendering
        newMetaJoints[jointName] = {
          ...newMetaJoints[jointName] || { color: '', zlevel: 0 },
          x: newPosition[0],
          y: newPosition[1]
        }
      }
    }

    // Clear trajectory and trigger auto-simulation if enabled
    clearTrajectory()
    triggerMechanismChange()

    setPylinkDoc({
      ...pylinkDoc,
      pylinkage: {
        ...pylinkDoc.pylinkage,
        joints: newJoints
      },
      meta: {
        ...pylinkDoc.meta,
        joints: newMetaJoints
      }
    })
  }, [pylinkDoc, getJointPosition])

  // ═══════════════════════════════════════════════════════════════════════════════
  // RIGID BODY TRANSLATION
  // ═══════════════════════════════════════════════════════════════════════════════
  // Moves a group of joints to new positions based on original positions + delta.
  // IMPORTANT: This does NOT recalculate distances or angles - it preserves the
  // exact structure of the mechanism and only applies a uniform translation.
  //
  // For Static joints:   set x, y to targetPosition (directly in pylinkage data)
  // For Crank joints:    set meta x, y to targetPosition (preserves distance and angle)
  // For Revolute joints: set meta x, y to targetPosition (preserves distance0 and distance1)
  //
  // @param jointNames - Array of joint names to move
  // @param originalPositions - Map of joint names to their original positions before drag started
  // @param dx - Delta X from drag start
  // @param dy - Delta Y from drag start
  // ═══════════════════════════════════════════════════════════════════════════════
  const translateGroupRigid = useCallback((
    jointNames: string[],
    originalPositions: Record<string, [number, number]>,
    dx: number,
    dy: number
  ) => {
    if (jointNames.length === 0) return

    const newJoints = [...pylinkDoc.pylinkage.joints]
    const newMetaJoints = { ...pylinkDoc.meta.joints }

    for (const jointName of jointNames) {
      const originalPos = originalPositions[jointName]
      if (!originalPos) continue

      const targetX = originalPos[0] + dx
      const targetY = originalPos[1] + dy

      const jointIndex = newJoints.findIndex(j => j.name === jointName)
      if (jointIndex === -1) continue

      const currentJoint = newJoints[jointIndex]

      if (currentJoint.type === 'Static') {
        // Static joints: set x, y directly in pylinkage data
        newJoints[jointIndex] = {
          type: 'Static',
          name: jointName,
          x: targetX,
          y: targetY
        } as StaticJoint
      } else if (currentJoint.type === 'Crank') {
        // Crank joints: set meta position, preserve distance and angle
        // Since all joints in the group move together, relative distances are preserved
        const currentMeta = newMetaJoints[jointName] || { color: '', zlevel: 0 }
        newMetaJoints[jointName] = {
          ...currentMeta,
          x: targetX,
          y: targetY
        }
        // Joint structure (distance, angle, joint0 ref) stays unchanged
      } else if (currentJoint.type === 'Revolute') {
        // Revolute joints: set meta position, preserve distances
        // Since all joints in the group move together, relative distances are preserved
        const currentMeta = newMetaJoints[jointName] || { color: '', zlevel: 0 }
        newMetaJoints[jointName] = {
          ...currentMeta,
          x: targetX,
          y: targetY
        }
        // Joint structure (distance0, distance1, joint0, joint1 refs) stays unchanged
      }
    }

    setPylinkDoc(prev => ({
      ...prev,
      pylinkage: {
        ...prev.pylinkage,
        joints: newJoints
      },
      meta: {
        ...prev.meta,
        joints: newMetaJoints
      }
    }))
  }, [pylinkDoc.pylinkage.joints, pylinkDoc.meta.joints])

  // Merge two joints together (source is absorbed into target)
  const mergeJoints = useCallback((sourceJoint: string, targetJoint: string) => {
    const result = calculateMergeResult(sourceJoint, targetJoint, pylinkDoc.meta.links)

    // Update links to point to target instead of source
    const newLinks = { ...pylinkDoc.meta.links }
    for (const update of result.linksToUpdate) {
      newLinks[update.linkName] = {
        ...newLinks[update.linkName],
        connects: update.newConnects
      }
    }

    // Delete redundant links (self-loops and duplicates)
    for (const linkName of result.linksToDelete) {
      delete newLinks[linkName]
    }

    // Remove the source joint
    const newPylinkageJoints = pylinkDoc.pylinkage.joints.filter(j => j.name !== sourceJoint)
    const newSolveOrder = pylinkDoc.pylinkage.solve_order.filter(name => name !== sourceJoint)
    const newJointsMeta = { ...pylinkDoc.meta.joints }
    delete newJointsMeta[sourceJoint]

    // Clear trajectory and trigger auto-simulation
    clearTrajectory()
    triggerMechanismChange()

    setPylinkDoc({
      ...pylinkDoc,
      pylinkage: {
        ...pylinkDoc.pylinkage,
        joints: newPylinkageJoints,
        solve_order: newSolveOrder
      },
      meta: {
        joints: newJointsMeta,
        links: newLinks
      }
    })

    setSelectedJoints([targetJoint])
    showStatus(`Merged ${sourceJoint} into ${targetJoint}`, 'success', 2500)
  }, [pylinkDoc, showStatus, triggerMechanismChange])

  // Create a new link between two points/joints
  // Note: This creates Static joints. For Revolute default, use createLinkWithRevoluteDefault
  const _createLink = useCallback((
    startPoint: [number, number],
    endPoint: [number, number],
    startJointName: string | null,
    endJointName: string | null
  ) => {
    const existingJointNames = pylinkDoc.pylinkage.joints.map(j => j.name)
    const existingLinkNames = Object.keys(pylinkDoc.meta.links)

    let actualStartJoint = startJointName
    let actualEndJoint = endJointName

    // Create start joint if needed
    if (!actualStartJoint) {
      actualStartJoint = generateJointName(existingJointNames)
      const newJoint: StaticJoint = {
        type: 'Static',
        name: actualStartJoint,
        x: startPoint[0],
        y: startPoint[1]
      }
      pylinkDoc.pylinkage.joints.push(newJoint)
      existingJointNames.push(actualStartJoint)
    }

    // Create end joint if needed
    if (!actualEndJoint) {
      actualEndJoint = generateJointName(existingJointNames)
      const newJoint: StaticJoint = {
        type: 'Static',
        name: actualEndJoint,
        x: endPoint[0],
        y: endPoint[1]
      }
      pylinkDoc.pylinkage.joints.push(newJoint)
    }

    // Create the link
    const linkName = generateLinkName(existingLinkNames)
    const linkColor = getDefaultColor(existingLinkNames.length)

    const newMeta = { ...pylinkDoc.meta }
    newMeta.links[linkName] = {
      color: linkColor,
      connects: [actualStartJoint, actualEndJoint]
    }

    // Update solve order
    const newSolveOrder = [...pylinkDoc.pylinkage.solve_order]
    if (!newSolveOrder.includes(actualStartJoint)) {
      newSolveOrder.push(actualStartJoint)
    }
    if (!newSolveOrder.includes(actualEndJoint)) {
      newSolveOrder.push(actualEndJoint)
    }

    // Clear trajectory and trigger auto-simulation
    clearTrajectory()
    triggerMechanismChange()

    setPylinkDoc({
      ...pylinkDoc,
      pylinkage: {
        ...pylinkDoc.pylinkage,
        joints: [...pylinkDoc.pylinkage.joints],
        solve_order: newSolveOrder
      },
      meta: newMeta
    })

    const length = calculateDistance(startPoint, endPoint)
    showStatus(`Created ${linkName} (${length.toFixed(1)} units)`, 'success', 2500)

    return linkName
  }, [pylinkDoc, showStatus, triggerMechanismChange])

  // Helper to find joints connected to a given joint via existing links
  const findConnectedJoints = useCallback((jointName: string): string[] => {
    const connected: string[] = []
    Object.values(pylinkDoc.meta.links).forEach(link => {
      if (link.connects.includes(jointName)) {
        link.connects.forEach(j => {
          if (j !== jointName && !connected.includes(j)) {
            connected.push(j)
          }
        })
      }
    })
    return connected
  }, [pylinkDoc.meta.links])

  // Helper to get joint position for link creation
  // Uses the existing getJointPosition function which handles all joint types
  const getJointPositionForCreate = useCallback((jointName: string): [number, number] | null => {
    return getJointPosition(jointName)
  }, [getJointPosition])

  // Create a new link between two points/joints
  // If user clicked on an existing joint, use it. Otherwise create a new joint.
  // New joints are Revolute if they connect to a non-static joint that has other connections,
  // otherwise Static.
  const createLinkWithRevoluteDefault = useCallback((
    startPoint: [number, number],
    endPoint: [number, number],
    startJointName: string | null,  // Only set if user clicked on an existing joint
    endJointName: string | null      // Only set if user clicked on an existing joint
  ) => {
    const existingJointNames = pylinkDoc.pylinkage.joints.map(j => j.name)
    const existingLinkNames = Object.keys(pylinkDoc.meta.links)
    let newJoints = [...pylinkDoc.pylinkage.joints]
    const newMetaJoints = { ...pylinkDoc.meta.joints }

    let actualStartJoint = startJointName
    let actualEndJoint = endJointName

    // Create start joint if user didn't click on an existing joint
    if (!actualStartJoint) {
      actualStartJoint = generateJointName(existingJointNames)

      // Check if we can make this a Revolute joint:
      // - The other end (endJointName) must be an existing joint that user clicked on
      // - That joint must have connections to other joints (via existing links)
      // - At least one of those connected joints should be usable as a reference
      let madeRevolute = false

      if (endJointName) {
        // User clicked on an existing joint for the end - check if we can make start a Revolute
        const endJoint = pylinkDoc.pylinkage.joints.find(j => j.name === endJointName)

        if (endJoint && endJoint.type !== 'Static') {
          // End joint is non-static, find another joint connected to it
          const connectedToEnd = findConnectedJoints(endJointName)

          // Find a suitable second reference joint (preferably non-static, or any if needed)
          const secondRef = connectedToEnd.find(j => {
            const joint = pylinkDoc.pylinkage.joints.find(jj => jj.name === j)
            return joint && joint.type !== 'Static'
          }) || connectedToEnd[0]

          if (secondRef) {
            const endPos = getJointPositionForCreate(endJointName)
            const secondPos = getJointPositionForCreate(secondRef)

            if (endPos && secondPos) {
              const distance0 = calculateDistance(endPos, startPoint)
              const distance1 = calculateDistance(secondPos, startPoint)

              const newJoint: RevoluteJoint = {
                type: 'Revolute',
                name: actualStartJoint,
                joint0: { ref: endJointName },
                joint1: { ref: secondRef },
                distance0,
                distance1
              }
              newJoints.push(newJoint)
              newMetaJoints[actualStartJoint] = { color: '', zlevel: 0, x: startPoint[0], y: startPoint[1], show_path: true }
              madeRevolute = true
            }
          }
        }
      }

      // If we couldn't make a Revolute, create as Static
      if (!madeRevolute) {
        const newJoint: StaticJoint = {
          type: 'Static',
          name: actualStartJoint,
          x: startPoint[0],
          y: startPoint[1]
        }
        newJoints.push(newJoint)
      }
      existingJointNames.push(actualStartJoint)
    }

    // Create end joint if user didn't click on an existing joint
    if (!actualEndJoint) {
      actualEndJoint = generateJointName(existingJointNames)

      // Check if we can make this a Revolute joint:
      // - The other end (actualStartJoint) must be an existing joint that user clicked on
      //   OR a joint we just created that connects to the mechanism
      // - That joint must have connections to other joints
      let madeRevolute = false

      if (startJointName) {
        // User clicked on an existing joint for the start - check if we can make end a Revolute
        const startJoint = pylinkDoc.pylinkage.joints.find(j => j.name === startJointName)

        if (startJoint && startJoint.type !== 'Static') {
          // Start joint is non-static, find another joint connected to it
          const connectedToStart = findConnectedJoints(startJointName)

          // Find a suitable second reference joint
          const secondRef = connectedToStart.find(j => {
            const joint = pylinkDoc.pylinkage.joints.find(jj => jj.name === j)
            return joint && joint.type !== 'Static'
          }) || connectedToStart[0]

          if (secondRef) {
            const startPos = getJointPositionForCreate(startJointName)
            const secondPos = getJointPositionForCreate(secondRef)

            if (startPos && secondPos) {
              const distance0 = calculateDistance(startPos, endPoint)
              const distance1 = calculateDistance(secondPos, endPoint)

              const newJoint: RevoluteJoint = {
                type: 'Revolute',
                name: actualEndJoint,
                joint0: { ref: startJointName },
                joint1: { ref: secondRef },
                distance0,
                distance1
              }
              newJoints.push(newJoint)
              newMetaJoints[actualEndJoint] = { color: '', zlevel: 0, x: endPoint[0], y: endPoint[1], show_path: true }
              madeRevolute = true
            }
          }
        }
      }

      // If we couldn't make a Revolute, create as Static
      if (!madeRevolute) {
        const newJoint: StaticJoint = {
          type: 'Static',
          name: actualEndJoint,
          x: endPoint[0],
          y: endPoint[1]
        }
        newJoints.push(newJoint)
      }
    }

    // IMPORTANT: Handle case where both joints already exist - ensure kinematic constraint
    // If one joint is Static and one is kinematic (Crank/Revolute), we need to either:
    // 1. Convert the Static to a Revolute (making it move with the mechanism)
    // 2. Or warn that this creates an over-constrained mechanism
    if (startJointName && endJointName) {
      const startJoint = newJoints.find(j => j.name === startJointName)
      const endJoint = newJoints.find(j => j.name === endJointName)

      if (startJoint && endJoint) {
        const startIsStatic = startJoint.type === 'Static'
        const endIsStatic = endJoint.type === 'Static'
        const startIsKinematic = startJoint.type === 'Crank' || startJoint.type === 'Revolute'
        const endIsKinematic = endJoint.type === 'Crank' || endJoint.type === 'Revolute'

        // Case: Static joint connected to kinematic joint - convert Static to Revolute
        if ((startIsStatic && endIsKinematic) || (endIsStatic && startIsKinematic)) {
          const staticJointName = startIsStatic ? startJointName : endJointName
          const kinematicJointName = startIsStatic ? endJointName : startJointName
          const staticJoint = startIsStatic ? startJoint : endJoint

          // Find another joint connected to the kinematic joint to serve as second reference
          const connectedToKinematic = findConnectedJoints(kinematicJointName)
          const secondRef = connectedToKinematic.find(j => j !== staticJointName) ||
                          // Or use any joint from the mechanism
                          pylinkDoc.pylinkage.joints.find(j =>
                            j.name !== staticJointName &&
                            j.name !== kinematicJointName &&
                            (j.type === 'Static' || j.type === 'Crank' || j.type === 'Revolute')
                          )?.name

          if (secondRef) {
            const kinematicPos = getJointPositionForCreate(kinematicJointName)
            const secondPos = getJointPositionForCreate(secondRef)
            const staticPos: [number, number] = [
              (staticJoint as StaticJoint).x,
              (staticJoint as StaticJoint).y
            ]

            if (kinematicPos && secondPos) {
              const distance0 = calculateDistance(kinematicPos, staticPos)
              const distance1 = calculateDistance(secondPos, staticPos)

              // Convert Static to Revolute
              const newRevoluteJoint: RevoluteJoint = {
                type: 'Revolute',
                name: staticJointName,
                joint0: { ref: kinematicJointName },
                joint1: { ref: secondRef },
                distance0,
                distance1
              }

              // Replace the Static joint with the Revolute
              const staticIndex = newJoints.findIndex(j => j.name === staticJointName)
              if (staticIndex >= 0) {
                newJoints[staticIndex] = newRevoluteJoint
                // Add to meta if not already there
                if (!newMetaJoints[staticJointName]) {
                  newMetaJoints[staticJointName] = {
                    color: '',
                    zlevel: 0,
                    x: staticPos[0],
                    y: staticPos[1],
                    show_path: true
                  }
                }
                showStatus(`Converted ${staticJointName} to Revolute joint`, 'info', 2000)
              }
            }
          } else {
            // Can't find second reference - warn user
            showStatus(
              `⚠️ Warning: Link connects kinematic to static joint without proper constraint. Mechanism may be invalid.`,
              'warning',
              4000
            )
          }
        }
      }
    }

    // Create the link
    const linkName = generateLinkName(existingLinkNames)
    const linkColor = getDefaultColor(existingLinkNames.length)

    const newMetaLinks = { ...pylinkDoc.meta.links }
    newMetaLinks[linkName] = {
      color: linkColor,
      connects: [actualStartJoint, actualEndJoint]
    }

    // Update solve order
    const newSolveOrder = [...pylinkDoc.pylinkage.solve_order]
    if (!newSolveOrder.includes(actualStartJoint)) {
      newSolveOrder.push(actualStartJoint)
    }
    if (!newSolveOrder.includes(actualEndJoint)) {
      newSolveOrder.push(actualEndJoint)
    }

    // Clear trajectory and trigger auto-simulation
    clearTrajectory()
    triggerMechanismChange()

    setPylinkDoc({
      ...pylinkDoc,
      pylinkage: {
        ...pylinkDoc.pylinkage,
        joints: newJoints,
        solve_order: newSolveOrder
      },
      meta: {
        ...pylinkDoc.meta,
        joints: newMetaJoints,
        links: newMetaLinks
      }
    })

    const length = calculateDistance(startPoint, endPoint)
    showStatus(`Created ${linkName} (${length.toFixed(1)} units)`, 'success', 2500)

    return linkName
  }, [pylinkDoc, showStatus, triggerMechanismChange, findConnectedJoints, getJointPositionForCreate])

  // Handle mouse down on canvas (for drag start)
  const handleCanvasMouseDown = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)
    const clickPoint: [number, number] = [x, y]

    // Handle group select mode - start box selection
    if (toolMode === 'group_select') {
      setGroupSelectionState({
        isSelecting: true,
        startPoint: clickPoint,
        currentPoint: clickPoint
      })
      showStatus('Drag to select multiple elements', 'action')
      return
    }

    // Handle move group mode - start dragging
    if (moveGroupState.isActive && moveGroupState.joints.length > 0) {
      // Check multiple ways the user might be clicking on the selection:
      // 1. Directly on a joint in the move group
      // 2. On a link connecting joints in the move group
      // 3. Within the bounding box of the selection (more lenient)

      const jointsWithPositions = getJointsWithPositions()
      const linksWithPositions = getLinksWithPositions()
      const nearestJoint = findNearestJoint(clickPoint, jointsWithPositions)
      const nearestLink = findNearestLink(clickPoint, linksWithPositions)

      // Check 1: Click directly on a joint in the move group
      const clickedOnJoint = nearestJoint && moveGroupState.joints.includes(nearestJoint.name)

      // Check 2: Click on a link where BOTH endpoints are in the move group
      const clickedOnGroupLink = nearestLink && (() => {
        const linkMeta = pylinkDoc.meta.links[nearestLink.name]
        if (!linkMeta) return false
        return linkMeta.connects.every(j => moveGroupState.joints.includes(j))
      })()

      // Check 3: Click within the bounding box of all selected joints (with padding)
      const BBOX_PADDING = 0.5  // Units of padding around the bounding box
      const clickedInBoundingBox = (() => {
        const positions = moveGroupState.joints
          .map(jointName => getJointPosition(jointName))
          .filter((pos): pos is [number, number] => pos !== null)

        if (positions.length === 0) return false

        const minX = Math.min(...positions.map(p => p[0])) - BBOX_PADDING
        const maxX = Math.max(...positions.map(p => p[0])) + BBOX_PADDING
        const minY = Math.min(...positions.map(p => p[1])) - BBOX_PADDING
        const maxY = Math.max(...positions.map(p => p[1])) + BBOX_PADDING

        return clickPoint[0] >= minX && clickPoint[0] <= maxX &&
               clickPoint[1] >= minY && clickPoint[1] <= maxY
      })()

      // Check 4: Click on a selected drawn object
      const clickedOnDrawnObj = moveGroupState.drawnObjectIds.some(id => {
        const obj = drawnObjects.objects.find(o => o.id === id)
        if (!obj) return false
        const minX = Math.min(...obj.points.map(p => p[0]))
        const maxX = Math.max(...obj.points.map(p => p[0]))
        const minY = Math.min(...obj.points.map(p => p[1]))
        const maxY = Math.max(...obj.points.map(p => p[1]))
        return clickPoint[0] >= minX && clickPoint[0] <= maxX &&
               clickPoint[1] >= minY && clickPoint[1] <= maxY
      })

      if (clickedOnJoint || clickedOnGroupLink || clickedInBoundingBox || clickedOnDrawnObj) {
        setMoveGroupState(prev => ({
          ...prev,
          isDragging: true,
          dragStartPoint: clickPoint
        }))
        const itemCount = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
        showStatus(`Moving ${itemCount} items — drag to reposition`, 'action')
        return
      } else {
        // Clicked outside selection - exit move mode
        exitMoveGroupMode()
        return  // Important: return to prevent other handlers from running
      }
    }

    // Handle select mode - drag joints
    if (toolMode === 'select') {
    const jointsWithPositions = getJointsWithPositions()
    const nearestJoint = findNearestJoint(clickPoint, jointsWithPositions)

    if (nearestJoint) {
      // Start dragging
      setDragState({
        isDragging: true,
        draggedJoint: nearestJoint.name,
        dragStartPosition: nearestJoint.position,
        currentPosition: nearestJoint.position,
        mergeTarget: null,
        mergeProximity: Infinity
      })
      setSelectedJoints([nearestJoint.name])
      setSelectedLinks([])
      showStatus(`Dragging ${nearestJoint.name}`, 'action')
    }
    }
  }, [toolMode, selectedJoints, getJointsWithPositions, getJointPosition, showStatus])

  // Handle mouse move on canvas
  const handleCanvasMouseMove = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)
    const currentPoint: [number, number] = [x, y]

    // Handle group selection box preview
    if (groupSelectionState.isSelecting && groupSelectionState.startPoint) {
      setGroupSelectionState(prev => ({
        ...prev,
        currentPoint
      }))

      // Preview what would be selected
      const jointsWithPositions = getJointsWithPositions()
      const linksWithPositions = getLinksWithPositions()
      const box = {
        x1: groupSelectionState.startPoint[0],
        y1: groupSelectionState.startPoint[1],
        x2: currentPoint[0],
        y2: currentPoint[1]
      }
      const preview = findElementsInBox(box, jointsWithPositions, linksWithPositions)
      showStatus(`Selecting ${preview.joints.length} joints, ${preview.links.length} links`, 'action')
      return
    }

    // Handle link creation preview
    if (linkCreationState.isDrawing) {
      const jointsWithPositions = getJointsWithPositions()
      const nearestJoint = findNearestJoint(currentPoint, jointsWithPositions)
      const endPoint: [number, number] = nearestJoint?.position || currentPoint

      setPreviewLine({
        start: linkCreationState.startPoint!,
        end: endPoint
      })
      return
    }

    // Handle move group dragging (multiple joints and/or drawn objects)
    // Uses rigid body translation to preserve mechanism structure
    if (moveGroupState.isDragging && moveGroupState.dragStartPoint) {
      const dx = currentPoint[0] - moveGroupState.dragStartPoint[0]
      const dy = currentPoint[1] - moveGroupState.dragStartPoint[1]

      // Use rigid body translation - moves from ORIGINAL positions by delta
      // This preserves all distances and angles in the mechanism
      translateGroupRigid(
        moveGroupState.joints,
        moveGroupState.startPositions,  // Original positions before drag started
        dx,
        dy
      )

      // Move all drawn objects by the delta (from their original positions)
      if (moveGroupState.drawnObjectIds.length > 0) {
        setDrawnObjects(prev => ({
          ...prev,
          objects: prev.objects.map(obj => {
            if (moveGroupState.drawnObjectIds.includes(obj.id)) {
              const originalPoints = moveGroupState.drawnObjectStartPositions[obj.id]
              if (originalPoints) {
                return {
                  ...obj,
                  points: originalPoints.map(p => [p[0] + dx, p[1] + dy] as [number, number])
                }
              }
            }
            return obj
          })
        }))
      }

      const totalItems = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
      showStatus(`Moving ${totalItems} items (Δ${dx.toFixed(1)}, ${dy.toFixed(1)})`, 'action')
      return
    }

    // Handle dragging single joint
    if (dragState.isDragging && dragState.draggedJoint) {
      const jointsWithPositions = getJointsWithPositions()
      const mergeTarget = findMergeTarget(currentPoint, jointsWithPositions, dragState.draggedJoint, jointMergeRadius)

      if (mergeTarget) {
        // Near a merge target
        setDragState(prev => ({
          ...prev,
          currentPosition: currentPoint,
          mergeTarget: mergeTarget.name,
          mergeProximity: mergeTarget.distance
        }))
        showStatus(`Release to merge into ${mergeTarget.name}`, 'action')
      } else {
        // Just moving
        setDragState(prev => ({
          ...prev,
          currentPosition: currentPoint,
          mergeTarget: null,
          mergeProximity: Infinity
        }))
        showStatus(`Moving ${dragState.draggedJoint} to (${x.toFixed(1)}, ${y.toFixed(1)})`, 'action')
      }

      // Update the joint position in real-time
      moveJoint(dragState.draggedJoint, currentPoint)
    }
  }, [linkCreationState.isDrawing, linkCreationState.startPoint, dragState.isDragging, dragState.draggedJoint, groupSelectionState.isSelecting, groupSelectionState.startPoint, moveGroupState, getJointsWithPositions, getLinksWithPositions, moveJoint, showStatus])

  // Handle mouse up on canvas (for drag end)
  const handleCanvasMouseUp = useCallback(() => {
    // Handle group selection completion
    if (groupSelectionState.isSelecting && groupSelectionState.startPoint && groupSelectionState.currentPoint) {
      const jointsWithPositions = getJointsWithPositions()
      const linksWithPositions = getLinksWithPositions()
      const box = {
        x1: groupSelectionState.startPoint[0],
        y1: groupSelectionState.startPoint[1],
        x2: groupSelectionState.currentPoint[0],
        y2: groupSelectionState.currentPoint[1]
      }
      const selected = findElementsInBox(box, jointsWithPositions, linksWithPositions)

      // Also check for drawn objects in the box
      const drawnObjectsInBox = drawnObjects.objects.filter(obj => {
        // Check if any point of the object is in the box
        const minX = Math.min(box.x1, box.x2)
        const maxX = Math.max(box.x1, box.x2)
        const minY = Math.min(box.y1, box.y2)
        const maxY = Math.max(box.y1, box.y2)
        return obj.points.some(p => p[0] >= minX && p[0] <= maxX && p[1] >= minY && p[1] <= maxY)
      }).map(obj => obj.id)

      // Also include DrawnObjects merged with selected links
      const mergedDrawnObjects = drawnObjects.objects
        .filter(obj => obj.mergedLinkName && selected.links.includes(obj.mergedLinkName))
        .map(obj => obj.id)

      // Combine both - those in box and those merged with selected links
      const allSelectedDrawnObjects = [...new Set([...drawnObjectsInBox, ...mergedDrawnObjects])]

      setSelectedJoints(selected.joints)
      setSelectedLinks(selected.links)
      setDrawnObjects(prev => ({ ...prev, selectedIds: allSelectedDrawnObjects }))
      setGroupSelectionState(initialGroupSelectionState)

      if (selected.joints.length > 0 || selected.links.length > 0 || allSelectedDrawnObjects.length > 0) {
        // Enter move mode with all selected items
        enterMoveGroupMode(selected.joints, allSelectedDrawnObjects)
      } else {
        showStatus('No elements selected', 'info', 1500)
      }
      return
    }

    // Handle move group drag completion
    if (moveGroupState.isDragging) {
      const totalItems = moveGroupState.joints.length + moveGroupState.drawnObjectIds.length
      showStatus(`Moved ${totalItems} items`, 'success', 2000)
      // Reset drag state but keep move mode active
      setMoveGroupState(prev => ({
        ...prev,
        isDragging: false,
        dragStartPoint: null,
        // Update start positions to current positions for next drag
        startPositions: Object.fromEntries(
          prev.joints.map(jointName => {
            const pos = getJointPosition(jointName)
            return [jointName, pos || [0, 0]]
          })
        ),
        drawnObjectStartPositions: Object.fromEntries(
          prev.drawnObjectIds.map(id => {
            const obj = drawnObjects.objects.find(o => o.id === id)
            return [id, obj ? [...obj.points] : []]
          })
        )
      }))
      triggerMechanismChange()
      return
    }

    // Handle single joint drag completion
    if (!dragState.isDragging || !dragState.draggedJoint) return

    if (dragState.mergeTarget) {
      // Perform merge
      mergeJoints(dragState.draggedJoint, dragState.mergeTarget)
    } else if (dragState.currentPosition) {
      // Just moved - show final position
      showStatus(`Moved ${dragState.draggedJoint} to (${dragState.currentPosition[0].toFixed(1)}, ${dragState.currentPosition[1].toFixed(1)})`, 'success', 2000)
    }

    setDragState(initialDragState)
  }, [dragState, groupSelectionState, moveGroupState, drawnObjects.objects, getJointPosition, mergeJoints, getJointsWithPositions, getLinksWithPositions, enterMoveGroupMode, showStatus, triggerMechanismChange])

  // Batch delete multiple items at once (handles all deletions in a single state update)
  const batchDelete = useCallback((jointsToDelete: string[], linksToDelete: string[], drawnObjectsToDelete: string[] = []) => {
    // Calculate all items to remove including orphans
    let allLinksToDelete = new Set(linksToDelete)
    let allJointsToDelete = new Set(jointsToDelete)

    // For each joint being deleted, add its connected links
    jointsToDelete.forEach(jointName => {
      Object.entries(pylinkDoc.meta.links).forEach(([linkName, linkMeta]) => {
        if (linkMeta.connects.includes(jointName)) {
          allLinksToDelete.add(linkName)
        }
      })
    })

    // For each link being deleted, find orphaned joints
    const remainingLinks = Object.entries(pylinkDoc.meta.links).filter(
      ([name]) => !allLinksToDelete.has(name)
    )

    // Find joints that would become orphaned
    pylinkDoc.pylinkage.joints.forEach(joint => {
      if (allJointsToDelete.has(joint.name)) return

      const hasConnection = remainingLinks.some(([_, meta]) =>
        meta.connects.includes(joint.name)
      )

      // If originally selected for deletion via link, and now orphaned, delete it
      if (!hasConnection) {
        // Check if this joint was connected to any deleted link
        const wasConnected = [...allLinksToDelete].some(linkName => {
          const link = pylinkDoc.meta.links[linkName]
          return link && link.connects.includes(joint.name)
        })
        if (wasConnected) {
          allJointsToDelete.add(joint.name)
        }
      }
    })

    // Build new state
    const newLinks = Object.fromEntries(
      Object.entries(pylinkDoc.meta.links).filter(([name]) => !allLinksToDelete.has(name))
    )
    const newJoints = pylinkDoc.pylinkage.joints.filter(j => !allJointsToDelete.has(j.name))
    const newSolveOrder = pylinkDoc.pylinkage.solve_order.filter(name => !allJointsToDelete.has(name))
    const newMetaJoints = Object.fromEntries(
      Object.entries(pylinkDoc.meta.joints).filter(([name]) => !allJointsToDelete.has(name))
    )

    // Also delete DrawnObjects that are merged with any deleted link
    const allDrawnObjectsToDelete = new Set(drawnObjectsToDelete)
    drawnObjects.objects.forEach(obj => {
      if (obj.mergedLinkName && allLinksToDelete.has(obj.mergedLinkName)) {
        allDrawnObjectsToDelete.add(obj.id)
      }
    })

    // Remove drawn objects
    const newDrawnObjects = drawnObjects.objects.filter(obj => !allDrawnObjectsToDelete.has(obj.id))

    // Apply all changes at once
    setPylinkDoc({
      ...pylinkDoc,
      pylinkage: {
        ...pylinkDoc.pylinkage,
        joints: newJoints,
        solve_order: newSolveOrder
      },
      meta: {
        joints: newMetaJoints,
        links: newLinks
      }
    })

    if (allDrawnObjectsToDelete.size > 0) {
      setDrawnObjects(prev => ({
        ...prev,
        objects: newDrawnObjects,
        selectedIds: prev.selectedIds.filter(id => !allDrawnObjectsToDelete.has(id))
      }))
    }

    // Clear selections and trigger update
    setSelectedJoints([])
    setSelectedLinks([])
    clearTrajectory()
    triggerMechanismChange()

    // Exit move mode if active
    if (moveGroupState.isActive) {
      setMoveGroupState(initialMoveGroupState)
    }

    return {
      deletedJoints: allJointsToDelete.size,
      deletedLinks: allLinksToDelete.size,
      deletedDrawnObjects: allDrawnObjectsToDelete.size
    }
  }, [pylinkDoc, drawnObjects.objects, moveGroupState.isActive, triggerMechanismChange])

  // Handle delete with confirmation for multiple items
  const handleDeleteSelected = useCallback(() => {
    const totalItems = selectedJoints.length + selectedLinks.length + drawnObjects.selectedIds.length

    if (totalItems === 0) {
      showStatus('Nothing selected to delete', 'info', 1500)
      return
    }

    if (totalItems > 1) {
      // Show confirmation dialog for multiple items
      setDeleteConfirmDialog({
        open: true,
        joints: selectedJoints,
        links: selectedLinks
      })
    } else {
      // Single item - delete directly
      const result = batchDelete(selectedJoints, selectedLinks, drawnObjects.selectedIds)
      showStatus(`Deleted ${result.deletedJoints} joints, ${result.deletedLinks} links`, 'success', 2500)
    }
  }, [selectedJoints, selectedLinks, drawnObjects.selectedIds, batchDelete, showStatus])

  // Confirm delete multiple items
  const confirmDelete = useCallback(() => {
    const { joints, links } = deleteConfirmDialog
    const drawnObjectIds = drawnObjects.selectedIds

    const result = batchDelete(joints, links, drawnObjectIds)

    setDeleteConfirmDialog({ open: false, joints: [], links: [] })
    showStatus(`Deleted ${result.deletedJoints} joints, ${result.deletedLinks} links${result.deletedDrawnObjects > 0 ? `, ${result.deletedDrawnObjects} objects` : ''}`, 'success', 2500)
  }, [deleteConfirmDialog, drawnObjects.selectedIds, batchDelete, showStatus])

  // Complete path drawing (called by Enter key or double-click)
  const completePathDrawing = useCallback(() => {
    if (pathDrawState.isDrawing && pathDrawState.points.length >= 2) {
      const newPath = createTargetPath(pathDrawState.points, targetPaths)
      setTargetPaths(prev => [...prev, newPath])
      setSelectedPathId(newPath.id)
      setPathDrawState(initialPathDrawState)
      showStatus(`Created target path with ${pathDrawState.points.length} points`, 'success', 2500)
    } else if (pathDrawState.isDrawing) {
      showStatus('Path needs at least 2 points', 'warning', 2000)
    }
  }, [pathDrawState, targetPaths, showStatus])

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return
      }

      // Escape to cancel
      if (event.key === 'Escape') {
        cancelAction()
        return
      }

      // Enter to complete path drawing
      if (event.key === 'Enter' && pathDrawState.isDrawing) {
        completePathDrawing()
        event.preventDefault()
        return
      }

      // Spacebar to play/pause animation
      if (event.key === ' ' || event.code === 'Space') {
        event.preventDefault()  // Prevent page scroll
        if (animationState.isAnimating) {
          pauseAnimation()
          showStatus('Animation paused', 'info', 1500)
        } else if (trajectoryData && trajectoryData.nSteps > 0) {
          playAnimation()
          showStatus('Animation playing', 'info', 1500)
        } else if (canSimulate(pylinkDoc.pylinkage.joints)) {
          // No trajectory yet, run simulation first then play
          runSimulation().then(() => {
            setTimeout(() => playAnimation(), 100)
          })
          showStatus('Running simulation...', 'action')
        }
        return
      }

      // Delete/Backspace/X to delete selected items (with confirmation for multiple)
      // Now also handles DrawnObjects (polygons)
      const hasSelectedItems = selectedJoints.length > 0 || selectedLinks.length > 0 || drawnObjects.selectedIds.length > 0
      if ((event.key === 'Delete' || event.key === 'Backspace' || event.key === 'x' || event.key === 'X') && hasSelectedItems) {
        // If X is pressed and it's not the shortcut intent (no modifier), delete selected
        if (event.key === 'x' || event.key === 'X') {
          // X is also the delete tool shortcut, so only delete if items are selected
          // and switch to delete mode if nothing selected
          if (!hasSelectedItems) {
            // Let it fall through to tool selection
          } else {
            handleDeleteSelected()
            event.preventDefault()
            return
          }
        } else {
          handleDeleteSelected()
          event.preventDefault()
          return
        }
      }

      const key = event.key.toUpperCase()
      const tool = TOOLS.find(t => t.shortcut === key)
      if (tool) {
        // If switching away from draw_link while drawing, cancel
        if (linkCreationState.isDrawing && tool.id !== 'draw_link') {
          setLinkCreationState(initialLinkCreationState)
          setPreviewLine(null)
        }
        // Cancel group selection if switching tools
        if (groupSelectionState.isSelecting && tool.id !== 'group_select') {
          setGroupSelectionState(initialGroupSelectionState)
        }
        // Cancel polygon drawing if switching tools
        if (polygonDrawState.isDrawing && tool.id !== 'draw_polygon') {
          setPolygonDrawState(initialPolygonDrawState)
        }
        // Cancel measurement if switching tools
        if (measureState.isMeasuring && tool.id !== 'measure') {
          setMeasureState(initialMeasureState)
        }
        // Cancel merge if switching tools
        if (mergePolygonState.step !== 'idle' && tool.id !== 'merge') {
          setMergePolygonState(initialMergePolygonState)
          setDrawnObjects(prev => ({ ...prev, selectedIds: [] }))
          setSelectedLinks([])
        }
        // Cancel path drawing if switching tools
        if (pathDrawState.isDrawing && tool.id !== 'draw_path') {
          setPathDrawState(initialPathDrawState)
        }
        setToolMode(tool.id)

        // Show appropriate message for merge mode
        if (tool.id === 'merge') {
          setMergePolygonState({ step: 'awaiting_selection', selectedPolygonId: null, selectedLinkName: null })
          showStatus('Select a link or a polygon to begin merge', 'action')
        } else if (tool.id === 'draw_path') {
          showStatus('Click to start drawing target path', 'action')
        } else {
          showStatus(`${tool.label} mode`, 'info', 1500)
        }
        event.preventDefault()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [linkCreationState.isDrawing, groupSelectionState.isSelecting, polygonDrawState.isDrawing, measureState.isMeasuring, pathDrawState.isDrawing, cancelAction, showStatus, selectedJoints, selectedLinks, handleDeleteSelected, animationState.isAnimating, playAnimation, pauseAnimation, trajectoryData, runSimulation, pylinkDoc.pylinkage.joints, completePathDrawing])

  // Handle canvas click
  const handleCanvasClick = useCallback((event: React.MouseEvent<SVGSVGElement>) => {
    // Don't process click if we just finished dragging or group selecting
    if (dragState.isDragging || groupSelectionState.isSelecting) return

    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const pixelX = event.clientX - rect.left
    const pixelY = event.clientY - rect.top
    const x = pixelsToUnits(pixelX)
    const y = pixelsToUnits(pixelY)
    const clickPoint: [number, number] = [x, y]

    const jointsWithPositions = getJointsWithPositions()
    const linksWithPositions = getLinksWithPositions()
    const nearestJoint = findNearestJoint(clickPoint, jointsWithPositions)
    const nearestLink = findNearestLink(clickPoint, linksWithPositions)

    // Handle mechanism select mode - select and enter move mode
    if (toolMode === 'mechanism_select') {
      // Helper to find DrawnObjects merged with mechanism links
      const findMergedDrawnObjects = (linkNames: string[]): string[] => {
        return drawnObjects.objects
          .filter(obj => obj.mergedLinkName && linkNames.includes(obj.mergedLinkName))
          .map(obj => obj.id)
      }

      if (nearestJoint) {
        const mechanism = findConnectedMechanism(nearestJoint.name, pylinkDoc.meta.links)
        setSelectedJoints(mechanism.joints)
        setSelectedLinks(mechanism.links)
        // Find DrawnObjects merged with this mechanism's links
        const mergedDrawnObjects = findMergedDrawnObjects(mechanism.links)
        setDrawnObjects(prev => ({ ...prev, selectedIds: mergedDrawnObjects }))
        // Enter move mode with the selected mechanism and merged DrawnObjects
        enterMoveGroupMode(mechanism.joints, mergedDrawnObjects)
      } else if (nearestLink) {
        // Get joints from the link and find their connected mechanism
        const linkMeta = pylinkDoc.meta.links[nearestLink.name]
        if (linkMeta && linkMeta.connects.length > 0) {
          const mechanism = findConnectedMechanism(linkMeta.connects[0], pylinkDoc.meta.links)
          setSelectedJoints(mechanism.joints)
          setSelectedLinks(mechanism.links)
          // Find DrawnObjects merged with this mechanism's links
          const mergedDrawnObjects = findMergedDrawnObjects(mechanism.links)
          setDrawnObjects(prev => ({ ...prev, selectedIds: mergedDrawnObjects }))
          // Enter move mode with the selected mechanism and merged DrawnObjects
          enterMoveGroupMode(mechanism.joints, mergedDrawnObjects)
        }
      } else {
        setSelectedJoints([])
        setSelectedLinks([])
        setDrawnObjects(prev => ({ ...prev, selectedIds: [] }))
        exitMoveGroupMode()
        showStatus('Click on a joint or link to select its mechanism', 'info', 1500)
      }
      return
    }

    // Handle measure mode
    if (toolMode === 'measure') {
      // Snap to existing joint if within MERGE_THRESHOLD
      const snappedPoint: [number, number] = nearestJoint?.position || clickPoint
      const snappedX = snappedPoint[0]
      const snappedY = snappedPoint[1]
      const snappedToJoint = nearestJoint ? ` (${nearestJoint.name})` : ''

      if (!measureState.isMeasuring) {
        // First click - set start point
        setMeasureState({
          isMeasuring: true,
          startPoint: snappedPoint,
          endPoint: null,
          measurementId: Date.now()
        })
        // Add start marker
        setMeasurementMarkers(prev => [...prev, {
          id: Date.now(),
          point: snappedPoint,
          timestamp: Date.now()
        }])
        showStatus(`Start: (${snappedX.toFixed(1)}, ${snappedY.toFixed(1)})${snappedToJoint} — click second point`, 'action')
      } else {
        // Second click - complete measurement
        const startPoint = measureState.startPoint!
        const distance = calculateDistance(startPoint, snappedPoint)
        const dx = snappedPoint[0] - startPoint[0]
        const dy = snappedPoint[1] - startPoint[1]

        // Add end marker
        setMeasurementMarkers(prev => [...prev, {
          id: Date.now(),
          point: snappedPoint,
          timestamp: Date.now()
        }])

        showStatus(`Distance: ${distance.toFixed(2)} units (Δx: ${dx.toFixed(1)}, Δy: ${dy.toFixed(1)})`, 'success', 5000)

        // Reset measurement state
        setMeasureState(initialMeasureState)

        // Start fade timer for markers (3 seconds)
        setTimeout(() => {
          setMeasurementMarkers(prev => prev.filter(m => Date.now() - m.timestamp < 3000))
        }, 3000)
      }
      return
    }

    // Handle merge mode - merge polygon with enclosed link (either order works)
    if (toolMode === 'merge') {
      // Helper to check if click is on a polygon (inside or on the stroke/outline)
      const findClickedPolygon = () => {
        return drawnObjects.objects.find(obj => {
          if (obj.type !== 'polygon' || obj.points.length < 3) return false
          // Skip already-merged polygons
          if (obj.mergedLinkName) return false
          // Check if inside polygon
          if (isPointInPolygon(clickPoint, obj.points)) return true
          // Check if on the polygon's stroke (near any edge)
          for (let i = 0; i < obj.points.length; i++) {
            const p1 = obj.points[i]
            const p2 = obj.points[(i + 1) % obj.points.length]
            // Point-to-line-segment distance check
            const lineLen = calculateDistance(p1, p2)
            if (lineLen === 0) continue
            const t = Math.max(0, Math.min(1, ((clickPoint[0] - p1[0]) * (p2[0] - p1[0]) + (clickPoint[1] - p1[1]) * (p2[1] - p1[1])) / (lineLen * lineLen)))
            const projX = p1[0] + t * (p2[0] - p1[0])
            const projY = p1[1] + t * (p2[1] - p1[1])
            const dist = calculateDistance(clickPoint, [projX, projY])
            if (dist < 0.5) return true  // Within 0.5 units of edge
          }
          return false
        })
      }

      // Helper to complete the merge
      const completeMerge = (polygonId: string, linkName: string) => {
        const polygon = drawnObjects.objects.find(obj => obj.id === polygonId)
        const linkMeta = pylinkDoc.meta.links[linkName]

        if (!polygon || !linkMeta) {
          showStatus('Error: Could not find polygon or link', 'error', 2000)
          setMergePolygonState(initialMergePolygonState)
          return false
        }

        const startPos = getJointPosition(linkMeta.connects[0])
        const endPos = getJointPosition(linkMeta.connects[1])

        if (!startPos || !endPos) {
          showStatus('Error: Could not get link endpoint positions', 'error', 2000)
          return false
        }

        // Check if both endpoints are inside the polygon
        if (!areLinkEndpointsInPolygon(startPos, endPos, polygon.points)) {
          showStatus(`Link "${linkName}" endpoints are not inside the polygon. Both ends must be enclosed.`, 'warning', 3500)
          return false
        }

        // Success! Merge the polygon with the link
        const linkColor = linkMeta.color || getDefaultColor(0)

        setDrawnObjects(prev => ({
          ...prev,
          objects: prev.objects.map(obj => {
            if (obj.id === polygonId) {
              return {
                ...obj,
                mergedLinkName: linkName,
                fillColor: linkColor,
                fillOpacity: 0.25,  // Subtle shaded color
                strokeColor: linkColor
              }
            }
            return obj
          }),
          selectedIds: []
        }))

        showStatus(`Merged polygon "${polygon.name}" with link "${linkName}"`, 'success', 3000)
        setMergePolygonState(initialMergePolygonState)
        return true
      }

      // Initial state: waiting for user to select either a polygon or a link
      // PRIORITY: If clicking near a link, select the link (even if inside a polygon)
      // This allows selecting links that are inside polygons
      if (mergePolygonState.step === 'awaiting_selection') {
        // Check for link FIRST - prioritize link selection over polygon
        if (nearestLink) {
          // User clicked on a link first
          setMergePolygonState({
            step: 'link_selected',
            selectedPolygonId: null,
            selectedLinkName: nearestLink.name
          })
          setSelectedLinks([nearestLink.name])
          setDrawnObjects(prev => ({ ...prev, selectedIds: [] }))  // Clear polygon selection
          showStatus(`Selected link "${nearestLink.name}" — now click a polygon to merge`, 'action')
          return
        }

        // No link clicked - check for polygon
        const clickedPolygon = findClickedPolygon()
        if (clickedPolygon) {
          // User clicked on an unmerged polygon
          setMergePolygonState({
            step: 'polygon_selected',
            selectedPolygonId: clickedPolygon.id,
            selectedLinkName: null
          })
          setDrawnObjects(prev => ({ ...prev, selectedIds: [clickedPolygon.id] }))
          setSelectedLinks([])  // Clear link selection
          showStatus(`Selected polygon "${clickedPolygon.name}" — now click a link to merge`, 'action')
          return
        }

        // Clicked on empty space - check if they clicked on an already-merged polygon
        const mergedPolygon = drawnObjects.objects.find(obj => {
          if (obj.type !== 'polygon' || obj.points.length < 3) return false
          if (!obj.mergedLinkName) return false
          return isPointInPolygon(clickPoint, obj.points)
        })
        if (mergedPolygon) {
          showStatus(`Polygon "${mergedPolygon.name}" is already merged with link "${mergedPolygon.mergedLinkName}"`, 'info', 2500)
        } else {
          showStatus('Select a link or a polygon to begin merge', 'info', 2000)
        }
        return
      }

      // Polygon selected: waiting for link
      if (mergePolygonState.step === 'polygon_selected') {
        // Link takes priority - complete the merge
        if (nearestLink) {
          const success = completeMerge(mergePolygonState.selectedPolygonId!, nearestLink.name)
          if (!success) {
            // Merge failed (endpoints not in polygon) - keep selection, let user try another link
            showStatus('Link endpoints must be inside the polygon. Try another link.', 'warning', 2500)
          }
          return
        }

        // No link - check if user clicked a different polygon to switch
        const clickedPolygon = findClickedPolygon()
        if (clickedPolygon && clickedPolygon.id !== mergePolygonState.selectedPolygonId) {
          // User clicked a different polygon - switch selection
          setMergePolygonState({
            step: 'polygon_selected',
            selectedPolygonId: clickedPolygon.id,
            selectedLinkName: null
          })
          setDrawnObjects(prev => ({ ...prev, selectedIds: [clickedPolygon.id] }))
          showStatus(`Switched to polygon "${clickedPolygon.name}" — now click a link to merge`, 'action')
        } else {
          showStatus('Click a link to merge with the selected polygon', 'info', 2000)
        }
        return
      }

      // Link selected: waiting for polygon
      if (mergePolygonState.step === 'link_selected') {
        // First check if user clicked a different link to switch
        if (nearestLink && nearestLink.name !== mergePolygonState.selectedLinkName) {
          // User clicked a different link - switch selection
          setMergePolygonState({
            step: 'link_selected',
            selectedPolygonId: null,
            selectedLinkName: nearestLink.name
          })
          setSelectedLinks([nearestLink.name])
          setDrawnObjects(prev => ({ ...prev, selectedIds: [] }))
          showStatus(`Switched to link "${nearestLink.name}" — now click a polygon to merge`, 'action')
          return
        }

        // Check for polygon (clicking inside or on edge)
        const clickedPolygon = findClickedPolygon()
        if (clickedPolygon) {
          const success = completeMerge(clickedPolygon.id, mergePolygonState.selectedLinkName!)
          if (!success) {
            // Merge failed - keep selection, let user try another polygon
            showStatus('Link endpoints must be inside the polygon. Try another polygon.', 'warning', 2500)
          }
          return
        }

        showStatus('Click inside a polygon to merge with the selected link', 'info', 2000)
        return
      }
    }

    // Handle draw_polygon mode - creates DrawnObjects only (no joints/links)
    if (toolMode === 'draw_polygon') {
      // Just use click position directly, no snapping to joints for polygons
      const point = clickPoint

      if (!polygonDrawState.isDrawing) {
        // First click - start polygon
        setPolygonDrawState({
          isDrawing: true,
          points: [point]
        })
        showStatus('Click to add polygon sides, click near start to close', 'action')
      } else {
        // Check if near starting point to close polygon
        const startPoint = polygonDrawState.points[0]
        const distanceToStart = calculateDistance(point, startPoint)

        if (distanceToStart <= MERGE_THRESHOLD && polygonDrawState.points.length >= 3) {
          // Close the polygon - create DrawnObject only (no joints/links)
          const polygonPoints = polygonDrawState.points
          const newDrawnObject = createDrawnObject(
            'polygon',
            polygonPoints,
            drawnObjects.objects.map(o => o.id)
          )
          setDrawnObjects(prev => ({
            ...prev,
            objects: [...prev.objects, newDrawnObject],
            selectedIds: [newDrawnObject.id]  // Select the new polygon
          }))

          setPolygonDrawState(initialPolygonDrawState)
          showStatus(`Completed ${polygonDrawState.points.length}-sided polygon (${newDrawnObject.id})`, 'success', 2500)
        } else {
          // Add new point to polygon
          setPolygonDrawState(prev => ({
            ...prev,
            points: [...prev.points, point]
          }))

          const sides = polygonDrawState.points.length
          showStatus(`${sides + 1} points — click near start to close polygon`, 'action')
        }
      }
      return
    }

    // Handle draw_path mode - for trajectory optimization target paths
    if (toolMode === 'draw_path') {
      const point = clickPoint

      if (!pathDrawState.isDrawing) {
        // First click - start path
        setPathDrawState({
          isDrawing: true,
          points: [point]
        })
        showStatus('Click to add points. Click near start or double-click to close path.', 'action')
      } else {
        // Check if clicking near start point to close the path (need at least 3 points)
        const startPoint = pathDrawState.points[0]
        const distanceToStart = calculateDistance(point, startPoint)

        if (distanceToStart <= jointMergeRadius && pathDrawState.points.length >= 3) {
          // Close the path - don't add the clicked point, just complete with existing points
          // The path is implicitly closed (last point connects back to first)
          const newPath = createTargetPath(pathDrawState.points, targetPaths)
          setTargetPaths(prev => [...prev, newPath])
          setSelectedPathId(newPath.id)
          setPathDrawState(initialPathDrawState)
          showStatus(`Created closed path with ${pathDrawState.points.length} points`, 'success', 2500)
        } else {
          // Add new point to path
          setPathDrawState(prev => ({
            ...prev,
            points: [...prev.points, point]
          }))

          const pointCount = pathDrawState.points.length + 1
          if (pointCount >= 3) {
            showStatus(`${pointCount} points — click near start to close, or double-click to finish`, 'action')
          } else {
            showStatus(`${pointCount} points — need at least 3 for a path`, 'action')
          }
        }
      }
      return
    }

    // Handle delete mode
    if (toolMode === 'delete') {
      // If items are selected, delete them
      if (selectedJoints.length > 0 || selectedLinks.length > 0) {
        handleDeleteSelected()
        return
      }

      // Otherwise, prioritize joints over links
      if (nearestJoint) {
        deleteJoint(nearestJoint.name)
      } else if (nearestLink) {
        deleteLink(nearestLink.name)
      } else {
        showStatus('Click on a joint or link to delete it', 'info', 2000)
      }
      return
    }

    // Handle draw_link mode
    if (toolMode === 'draw_link') {
      if (!linkCreationState.isDrawing) {
        // First click - start the link
        const startJointName = nearestJoint?.name || null
        const startPoint = nearestJoint?.position || clickPoint

        setLinkCreationState({
          isDrawing: true,
          startPoint,
          startJointName,
          endPoint: null
        })

        if (startJointName) {
          showStatus(`Drawing from ${startJointName} — click to complete`, 'action')
        } else {
          showStatus(`Drawing from (${startPoint[0].toFixed(1)}, ${startPoint[1].toFixed(1)}) — click to complete`, 'action')
        }
      } else {
        // Second click - complete the link
        const endJointName = nearestJoint?.name || null
        const endPoint = nearestJoint?.position || clickPoint

        // Don't allow connecting a joint to itself
        if (endJointName && endJointName === linkCreationState.startJointName) {
          showStatus('Cannot connect a joint to itself', 'warning', 2000)
          return
        }

        createLinkWithRevoluteDefault(
          linkCreationState.startPoint!,
          endPoint,
          linkCreationState.startJointName,
          endJointName
        )

        // Reset link creation state
        setLinkCreationState(initialLinkCreationState)
        setPreviewLine(null)
      }
      return
    }

    // Handle select mode (only for clicks, not drags)
    if (toolMode === 'select') {
      if (nearestJoint) {
        setSelectedJoints([nearestJoint.name])
        setSelectedLinks([])
        showStatus(`Selected ${nearestJoint.name}`, 'info', 1500)
      } else if (nearestLink) {
        setSelectedLinks([nearestLink.name])
        setSelectedJoints([])
        showStatus(`Selected ${nearestLink.name}`, 'info', 1500)
      } else {
        setSelectedJoints([])
        setSelectedLinks([])
        clearStatus()
      }
    }
  }, [toolMode, linkCreationState, dragState.isDragging, groupSelectionState.isSelecting, measureState, polygonDrawState, pathDrawState, selectedJoints, selectedLinks, getJointsWithPositions, getLinksWithPositions, pylinkDoc.meta.links, pylinkDoc.pylinkage.joints, deleteJoint, deleteLink, handleDeleteSelected, showStatus, clearStatus, triggerMechanismChange, createLinkWithRevoluteDefault])

  // Handle canvas double-click (for completing path drawing)
  const handleCanvasDoubleClick = useCallback((_event: React.MouseEvent<SVGSVGElement>) => {
    // Complete path drawing on double-click
    if (toolMode === 'draw_path' && pathDrawState.isDrawing && pathDrawState.points.length >= 2) {
      const newPath = createTargetPath(pathDrawState.points, targetPaths)
      setTargetPaths(prev => [...prev, newPath])
      setSelectedPathId(newPath.id)
      setPathDrawState(initialPathDrawState)
      showStatus(`Created target path with ${pathDrawState.points.length} points`, 'success', 2500)
      return
    }
  }, [toolMode, pathDrawState, targetPaths, showStatus])

  // Get cursor style based on tool mode
  const getCursorStyle = () => {
    if (moveGroupState.isDragging) return 'grabbing'
    if (moveGroupState.isActive) return 'move'
    if (dragState.isDragging) return 'grabbing'
    if (groupSelectionState.isSelecting) return 'crosshair'
    switch (toolMode) {
      case 'draw_link': return 'crosshair'
      case 'draw_polygon': return 'crosshair'
      case 'draw_path': return 'crosshair'
      case 'measure': return 'crosshair'
      case 'delete': return 'pointer'
      case 'select': return 'default'
      case 'group_select': return 'crosshair'
      case 'mechanism_select': return 'pointer'
      default: return 'default'
    }
  }

  // Render joints
  const renderJoints = () => {
    return pylinkDoc.pylinkage.joints.map((joint, index) => {
      const pos = getJointPosition(joint.name)
      if (!pos) return null

      const isSelected = selectedJoints.includes(joint.name)
      const isInMoveGroup = moveGroupState.isActive && moveGroupState.joints.includes(joint.name)
      const isHovered = hoveredJoint === joint.name
      const isDragging = dragState.draggedJoint === joint.name
      const isMergeTarget = dragState.mergeTarget === joint.name
      const meta = pylinkDoc.meta.joints[joint.name]
      const color = meta?.color || getDefaultColor(index)

      const isStatic = joint.type === 'Static'
      const isCrank = joint.type === 'Crank'

      // Highlight state flags (must be declared before using them)
      const showMergeHighlight = isMergeTarget || (isDragging && dragState.mergeTarget)
      const showMoveGroupHighlight = isInMoveGroup && !showMergeHighlight

      // Use theme colors for joint fill - keep original colors, selection indicated by glow
      const staticColor = jointColors.static      // #E74C3C - red for fixed joints
      const crankColor = jointColors.crank        // #F39C12 - amber for crank joints

      // Joint fill color based on type (NOT changed by selection - selection uses glow)
      const jointFillColor = isStatic
        ? staticColor
        : isCrank
          ? crankColor
          : color

      // Determine highlight type for glow effect
      const highlightType: HighlightType = showMergeHighlight
        ? 'merge'
        : showMoveGroupHighlight
          ? 'move_group'
          : (isSelected || isDragging)
            ? 'selected'
            : isHovered
              ? 'hovered'
              : 'none'

      // Get highlight styling - joints glow in their original color
      const defaultStroke = darkMode ? '#333333' : '#FFFFFF'
      const highlightStyle = getHighlightStyle('joint', highlightType, jointFillColor, 2)

      // Use the highlight filter, or no filter if not highlighted
      // Stroke color: use object color for glow, but keep default stroke for the actual outline
      const strokeColor = highlightType === 'none'
        ? defaultStroke
        : highlightType === 'move_group' || highlightType === 'merge'
          ? highlightStyle.stroke
          : defaultStroke  // Keep white/dark outline, the glow provides the color

      return (
        <g key={joint.name}>
          <circle
            cx={unitsToPixels(pos[0])}
            cy={unitsToPixels(pos[1])}
            r={isDragging ? jointSize * 1.5 : (isHovered || isSelected || isMergeTarget || isInMoveGroup ? jointSize * 1.25 : jointSize)}
            fill={jointFillColor}
            stroke={strokeColor}
            strokeWidth={highlightStyle.strokeWidth}
            filter={highlightStyle.filter}
            style={{ cursor: moveGroupState.isActive ? 'move' : (toolMode === 'select' ? 'grab' : 'pointer') }}
            onMouseEnter={() => !dragState.isDragging && !moveGroupState.isDragging && setHoveredJoint(joint.name)}
            onMouseLeave={() => setHoveredJoint(null)}
            onDoubleClick={(e) => {
              if (toolMode === 'select') {
                e.stopPropagation()
                openJointEditModal(joint.name)
              }
            }}
          />
          {/* Joint label - always show on hover/drag, or when showJointLabels is enabled */}
          {(showJointLabels || isHovered || isDragging || isSelected) && (
            <text
              x={unitsToPixels(pos[0])}
              y={unitsToPixels(pos[1]) - 14}
              textAnchor="middle"
              fontSize="11"
              fontWeight={isHovered || isDragging ? 'bold' : 'normal'}
              fill={showMergeHighlight ? jointColors.mergeHighlight : (darkMode ? '#e0e0e0' : '#333')}
              style={{ pointerEvents: 'none' }}
            >
              {joint.name}
            </text>
          )}
          {isStatic && !isDragging && (
            <rect
              x={unitsToPixels(pos[0]) - 4}
              y={unitsToPixels(pos[1]) + 10}
              width={8}
              height={4}
              fill="#e74c3c"
            />
          )}
          {isCrank && !isDragging && (
            <path
              d={`M ${unitsToPixels(pos[0])} ${unitsToPixels(pos[1]) + 10} L ${unitsToPixels(pos[0]) - 6} ${unitsToPixels(pos[1]) + 18} L ${unitsToPixels(pos[0]) + 6} ${unitsToPixels(pos[1]) + 18} Z`}
              fill="#f39c12"
            />
          )}
        </g>
      )
    })
  }

  // Render trajectory dots for simulation results
  const renderTrajectories = () => {
    if (!trajectoryData || !showTrajectory) return null

    // ─────────────────────────────────────────────────────────────────────────
    // TRAJECTORY COLOR CYCLE
    // ─────────────────────────────────────────────────────────────────────────
    // This function generates colors for trajectory dots that cycle back to
    // the starting color at the end of the animation. This creates a smooth
    // visual loop where the last dot's color nearly matches the first dot's.
    //
    // Available cycle types (configured in theme.ts):
    //   - 'rainbow': Full hue rotation through color wheel
    //   - 'fire':    Orange → dark → orange (warm, dramatic)
    //   - 'glow':    Orange → light/white → orange (bright, ethereal)
    //
    // The color at step 0 will be very close to the color at step (totalSteps-1),
    // ensuring trajectories that complete a full cycle look visually continuous.
    // ─────────────────────────────────────────────────────────────────────────
    // Use the user-configurable trajectory color cycle from Settings
    const getTrajectoryColor = (stepIndex: number, totalSteps: number) => {
      return getCyclicColor(stepIndex, totalSteps, trajectoryColorCycle)
    }

    return Object.entries(trajectoryData.trajectories).map(([jointName, positions]) => {
      // Skip if no trajectory data for this joint
      if (!positions || !Array.isArray(positions) || positions.length === 0) {
        return null
      }

      // Show trajectories for Revolute and Crank joints (they move during simulation)
      const jointType = trajectoryData.jointTypes?.[jointName]
      if (jointType !== 'Revolute' && jointType !== 'Crank') return null

      // Check if this joint has show_path enabled in meta (per-joint visibility control)
      const jointMeta = pylinkDoc.meta.joints[jointName]
      if (jointMeta && jointMeta.show_path === false) return null

      // Validate first position exists and has coordinates
      const firstPos = positions[0]
      if (!firstPos || typeof firstPos[0] !== 'number' || typeof firstPos[1] !== 'number') {
        return null
      }

      // Check if joint actually moves (has varying positions)
      const hasMovement = positions.length > 1 && positions.some((pos, i) =>
        i > 0 && pos && pos[0] !== undefined && pos[1] !== undefined &&
        (Math.abs(pos[0] - firstPos[0]) > 0.001 || Math.abs(pos[1] - firstPos[1]) > 0.001)
      )

      // Get color based on joint type for non-moving indicators
      const jointColor = jointType === 'Crank' ? jointColors.crank : jointColors.pivot

      // Filter out any invalid positions for rendering
      const validPositions = positions.filter(pos =>
        pos && typeof pos[0] === 'number' && typeof pos[1] === 'number' &&
        isFinite(pos[0]) && isFinite(pos[1])
      )

      if (validPositions.length === 0) return null

      return (
        <g key={`trajectory-${jointName}`}>
          {hasMovement && validPositions.length > 1 ? (
            <>
              {/* Draw trajectory path as a line (if trajectoryStyle includes line) */}
              {(trajectoryStyle === 'line' || trajectoryStyle === 'both') && (
                <path
                  d={validPositions
                    .filter(pos => pos && pos[0] !== undefined && pos[1] !== undefined)
                    .map((pos, i) =>
                      `${i === 0 ? 'M' : 'L'} ${unitsToPixels(pos[0])} ${unitsToPixels(pos[1])}`
                    ).join(' ')}
                  fill="none"
                  stroke="rgba(120, 120, 120, 0.5)"
                  strokeWidth={2}
                  strokeDasharray="4,2"
                />
              )}
              {/* Draw dots at each timestep (if trajectoryStyle includes dots) */}
              {(trajectoryStyle === 'dots' || trajectoryStyle === 'both') && validPositions.map((pos, stepIndex) => {
                // Extra safety check for each position
                if (!pos || pos[0] === undefined || pos[1] === undefined) return null
                return (
                  <circle
                    key={`${jointName}-step-${stepIndex}`}
                    cx={unitsToPixels(pos[0])}
                    cy={unitsToPixels(pos[1])}
                    r={trajectoryDotSize}
                    fill={getTrajectoryColor(stepIndex, validPositions.length)}
                    stroke="#fff"
                    strokeWidth={1}
                    opacity={0.8}
                  >
                    <title>{`${jointName} step ${stepIndex + 1}/${validPositions.length}: (${pos[0]?.toFixed(1) ?? '?'}, ${pos[1]?.toFixed(1) ?? '?'})`}</title>
                  </circle>
                )
              })}
            </>
          ) : (
            /* Non-moving joint: show a stationary indicator ring */
            <circle
              cx={unitsToPixels(validPositions[0][0])}
              cy={unitsToPixels(validPositions[0][1])}
              r={10}
              fill="none"
              stroke={jointColor}
              strokeWidth={2}
              strokeDasharray="3,3"
              opacity={0.6}
            >
              <title>{`${jointName}: stationary (not constrained to moving joints)`}</title>
            </circle>
          )}
        </g>
      )
    })
  }

  // Render links between joints
  const renderLinks = () => {
    const links: JSX.Element[] = []

    Object.entries(pylinkDoc.meta.links).forEach(([linkName, linkMeta], index) => {
      if (linkMeta.connects.length < 2) return

      const pos0 = getJointPosition(linkMeta.connects[0])
      const pos1 = getJointPosition(linkMeta.connects[1])

      if (!pos0 || !pos1) return

      const isHovered = hoveredLink === linkName
      const isSelected = selectedLinks.includes(linkName)
      // Check if link's joints are in move group
      const isInMoveGroup = moveGroupState.isActive &&
        linkMeta.connects.every(jointName => moveGroupState.joints.includes(jointName))

      // Check if this is a ground link
      const isGroundLink = linkMeta.isGround || false

      // Check if this link would stretch (invalid kinematic constraint)
      const isStretchingLink = stretchingLinks.includes(linkName)

      // Base link color - stretching links are red, ground links use gray if no custom color
      const baseLinkColor = isStretchingLink
        ? '#ff0000'  // Red for invalid/stretching links
        : isGroundLink && !linkMeta.color
          ? '#7f7f7f'  // Default gray for ground links
          : (linkMeta.color || getDefaultColor(index))

      // Determine highlight type for glow effect
      const linkHighlightType: HighlightType = isInMoveGroup
        ? 'move_group'
        : isSelected
          ? 'selected'
          : isHovered
            ? 'hovered'
            : 'none'

      // Get highlight styling - links glow in their original color
      const linkHighlightStyle = getHighlightStyle('link', linkHighlightType, baseLinkColor, linkThickness)

      // Ground links are rendered slightly thinner
      const effectiveStrokeWidth = isGroundLink
        ? Math.max(linkHighlightStyle.strokeWidth * 0.8, 2)
        : linkHighlightStyle.strokeWidth

      // Calculate midpoint for label
      const midX = (pos0[0] + pos1[0]) / 2
      const midY = (pos0[1] + pos1[1]) / 2

      links.push(
        <g key={linkName}>
          <line
            x1={unitsToPixels(pos0[0])}
            y1={unitsToPixels(pos0[1])}
            x2={unitsToPixels(pos1[0])}
            y2={unitsToPixels(pos1[1])}
            stroke={baseLinkColor}
            strokeWidth={effectiveStrokeWidth}
            strokeLinecap="round"
            strokeDasharray={isGroundLink ? '8,4' : undefined}  // Dashed for ground links
            filter={linkHighlightStyle.filter}
            style={{ cursor: moveGroupState.isActive ? 'move' : 'pointer' }}
            onMouseEnter={() => !moveGroupState.isDragging && setHoveredLink(linkName)}
            onMouseLeave={() => setHoveredLink(null)}
            onDoubleClick={(e) => {
              if (toolMode === 'select') {
                e.stopPropagation()
                openLinkEditModal(linkName)
              }
            }}
          />
          {/* Link label - show on hover/selected, or when showLinkLabels is enabled */}
          {(showLinkLabels || isHovered || isSelected) && (
            <g>
              {/* Background for readability */}
              <rect
                x={unitsToPixels(midX) - linkName.length * 3.5 - 4}
                y={unitsToPixels(midY) - 8}
                width={linkName.length * 7 + 8}
                height={14}
                fill={darkMode ? 'rgba(30, 30, 30, 0.85)' : 'rgba(255, 255, 255, 0.85)'}
                rx={3}
                style={{ pointerEvents: 'none' }}
              />
              <text
                x={unitsToPixels(midX)}
                y={unitsToPixels(midY) + 3}
                textAnchor="middle"
                fontSize="10"
                fontWeight={isHovered ? 'bold' : 'normal'}
                fill={isHovered || isSelected ? baseLinkColor : (darkMode ? '#b0b0b0' : '#555')}
                style={{ pointerEvents: 'none' }}
              >
                {linkName}
              </text>
            </g>
          )}
        </g>
      )
    })

    return links
  }

  // Render grid - now uses full canvas dimensions
  // Uses CSS variables for dark mode support
  const renderGrid = () => {
    const lines: JSX.Element[] = []
    const maxUnitsX = pixelsToUnits(canvasDimensions.width)
    const maxUnitsY = pixelsToUnits(canvasDimensions.height)

    // Get CSS variable values for grid colors (dark mode aware)
    const gridMajorColor = darkMode ? '#444444' : '#dddddd'
    const gridMinorColor = darkMode ? '#333333' : '#eeeeee'
    const gridTextColor = darkMode ? '#666666' : '#999999'

    // Major grid every 20 units
    for (let i = 0; i <= Math.ceil(maxUnitsX); i += 20) {
      lines.push(
        <line
          key={`v-major-${i}`}
          x1={unitsToPixels(i)}
          y1={0}
          x2={unitsToPixels(i)}
          y2={canvasDimensions.height}
          stroke={gridMajorColor}
          strokeWidth={1}
        />
      )
      if (i > 0) {
        lines.push(
          <text key={`vl-${i}`} x={unitsToPixels(i) + 2} y={12} fontSize="10" fill={gridTextColor}>
            {i}
          </text>
        )
      }
    }

    for (let i = 0; i <= Math.ceil(maxUnitsY); i += 20) {
      lines.push(
        <line
          key={`h-major-${i}`}
          x1={0}
          y1={unitsToPixels(i)}
          x2={canvasDimensions.width}
          y2={unitsToPixels(i)}
          stroke={gridMajorColor}
          strokeWidth={1}
        />
      )
      if (i > 0) {
        lines.push(
          <text key={`hl-${i}`} x={2} y={unitsToPixels(i) - 2} fontSize="10" fill={gridTextColor}>
            {i}
          </text>
        )
      }
    }

    // Minor grid every 10 units
    for (let i = 10; i <= Math.ceil(maxUnitsX); i += 20) {
      lines.push(
        <line
          key={`v-minor-${i}`}
          x1={unitsToPixels(i)}
          y1={0}
          x2={unitsToPixels(i)}
          y2={canvasDimensions.height}
          stroke={gridMinorColor}
          strokeWidth={0.5}
          strokeDasharray="2,4"
        />
      )
    }

    for (let i = 10; i <= Math.ceil(maxUnitsY); i += 20) {
      lines.push(
        <line
          key={`h-minor-${i}`}
          x1={0}
          y1={unitsToPixels(i)}
          x2={canvasDimensions.width}
          y2={unitsToPixels(i)}
          stroke={gridMinorColor}
          strokeWidth={0.5}
          strokeDasharray="2,4"
        />
      )
    }

    return lines
  }

  // Render preview line during link creation
  const renderPreviewLine = () => {
    if (!previewLine) return null

    return (
      <line
        x1={unitsToPixels(previewLine.start[0])}
        y1={unitsToPixels(previewLine.start[1])}
        x2={unitsToPixels(previewLine.end[0])}
        y2={unitsToPixels(previewLine.end[1])}
        stroke="#ff8c00"
        strokeWidth={3}
        strokeDasharray="8,4"
        strokeLinecap="round"
        opacity={0.7}
      />
    )
  }

  // Render group selection box (dashed rectangle)
  const renderSelectionBox = () => {
    if (!groupSelectionState.isSelecting || !groupSelectionState.startPoint || !groupSelectionState.currentPoint) {
      return null
    }

    const x1 = unitsToPixels(groupSelectionState.startPoint[0])
    const y1 = unitsToPixels(groupSelectionState.startPoint[1])
    const x2 = unitsToPixels(groupSelectionState.currentPoint[0])
    const y2 = unitsToPixels(groupSelectionState.currentPoint[1])

    const minX = Math.min(x1, x2)
    const minY = Math.min(y1, y2)
    const width = Math.abs(x2 - x1)
    const height = Math.abs(y2 - y1)

    return (
      <rect
        x={minX}
        y={minY}
        width={width}
        height={height}
        fill="rgba(25, 118, 210, 0.1)"
        stroke="#1976d2"
        strokeWidth={2}
        strokeDasharray="8,4"
        pointerEvents="none"
      />
    )
  }

  // Render completed drawn objects (polygons, shapes)
  const renderDrawnObjects = () => {
    if (drawnObjects.objects.length === 0) return null

    return drawnObjects.objects.map(obj => {
      if (obj.type === 'polygon' && obj.points.length >= 3) {
        const pathData = obj.points.map((p, i) =>
          `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`
        ).join(' ') + ' Z'

        const isSelected = drawnObjects.selectedIds.includes(obj.id)
        const isInMoveGroup = moveGroupState.isActive && moveGroupState.drawnObjectIds.includes(obj.id)

        // Determine highlight type for glow effect
        const polygonHighlightType: HighlightType = isInMoveGroup
          ? 'move_group'
          : isSelected
            ? 'selected'
            : 'none'

        // Get highlight styling - keeps original fill, adds glow outline in object's color
        const polygonHighlightStyle = getHighlightStyle('polygon', polygonHighlightType, obj.strokeColor, obj.strokeWidth)

        return (
          <g key={obj.id}>
            <path
              d={pathData}
              fill={obj.fillColor}  // Keep original fill color
              stroke={obj.strokeColor}  // Keep original stroke color
              strokeWidth={polygonHighlightStyle.strokeWidth}
              fillOpacity={obj.fillOpacity}
              filter={polygonHighlightStyle.filter}
              style={{ cursor: moveGroupState.isActive ? 'move' : 'pointer' }}
              onClick={(e) => {
                // In merge mode, let the canvas handler process this click
                // (don't stopPropagation, don't handle selection here)
                if (toolMode === 'merge') {
                  // Don't stop propagation - let the canvas click handler handle merge logic
                  return
                }

                e.stopPropagation()
                // Toggle selection (non-merge mode)
                setDrawnObjects(prev => ({
                  ...prev,
                  selectedIds: prev.selectedIds.includes(obj.id)
                    ? prev.selectedIds.filter(id => id !== obj.id)
                    : [...prev.selectedIds, obj.id]
                }))
              }}
            />
            {/* Show object name on hover/selection */}
            {isSelected && (
              <text
                x={unitsToPixels(obj.points[0][0])}
                y={unitsToPixels(obj.points[0][1]) - 8}
                fontSize="10"
                fill={obj.strokeColor}
                fontWeight="500"
              >
                {obj.name}
              </text>
            )}
          </g>
        )
      }
      return null
    })
  }

  // Render polygon preview during drawing
  const renderPolygonPreview = () => {
    if (!polygonDrawState.isDrawing || polygonDrawState.points.length === 0) return null

    const points = polygonDrawState.points

    // Draw lines between points
    const lines = []
    for (let i = 0; i < points.length - 1; i++) {
      lines.push(
        <line
          key={`poly-line-${i}`}
          x1={unitsToPixels(points[i][0])}
          y1={unitsToPixels(points[i][1])}
          x2={unitsToPixels(points[i + 1][0])}
          y2={unitsToPixels(points[i + 1][1])}
          stroke="#9c27b0"
          strokeWidth={3}
          strokeDasharray="6,3"
          opacity={0.8}
        />
      )
    }

    // Draw points
    const pointMarkers = points.map((point, i) => (
      <circle
        key={`poly-point-${i}`}
        cx={unitsToPixels(point[0])}
        cy={unitsToPixels(point[1])}
        r={i === 0 ? 10 : 6}
        fill={i === 0 ? '#9c27b0' : '#ce93d8'}
        stroke="#fff"
        strokeWidth={2}
        opacity={0.9}
      />
    ))

    // If we have 3+ points, show a preview fill
    let polygonFill = null
    if (points.length >= 3) {
      const pathData = points.map((p, i) =>
        `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`
      ).join(' ') + ' Z'

      polygonFill = (
        <path
          d={pathData}
          fill="rgba(156, 39, 176, 0.15)"
          stroke="none"
          pointerEvents="none"
        />
      )
    }

    return (
      <g>
        {polygonFill}
        {lines}
        {pointMarkers}
        {/* Show hint circle around starting point */}
        {points.length >= 3 && (
          <circle
            cx={unitsToPixels(points[0][0])}
            cy={unitsToPixels(points[0][1])}
            r={unitsToPixels(MERGE_THRESHOLD)}
            fill="none"
            stroke="#9c27b0"
            strokeWidth={1}
            strokeDasharray="4,4"
            opacity={0.5}
          />
        )}
      </g>
    )
  }

  // Render target path preview during drawing
  const renderPathPreview = () => {
    if (!pathDrawState.isDrawing || pathDrawState.points.length === 0) return null

    const points = pathDrawState.points
    const canClose = points.length >= 3

    // Draw lines between points
    const lines = []
    for (let i = 0; i < points.length - 1; i++) {
      lines.push(
        <line
          key={`path-line-${i}`}
          x1={unitsToPixels(points[i][0])}
          y1={unitsToPixels(points[i][1])}
          x2={unitsToPixels(points[i + 1][0])}
          y2={unitsToPixels(points[i + 1][1])}
          stroke="#e91e63"
          strokeWidth={3}
          strokeDasharray="8,4"
          opacity={0.9}
        />
      )
    }

    // Draw closing line preview (dashed, faded) - shows the path will be closed
    const closingLine = points.length >= 2 ? (
      <line
        key="path-closing-line"
        x1={unitsToPixels(points[points.length - 1][0])}
        y1={unitsToPixels(points[points.length - 1][1])}
        x2={unitsToPixels(points[0][0])}
        y2={unitsToPixels(points[0][1])}
        stroke="#e91e63"
        strokeWidth={2}
        strokeDasharray="4,8"
        opacity={0.4}
      />
    ) : null

    // Draw snap circle around start point when path can be closed
    const snapCircle = canClose ? (
      <circle
        cx={unitsToPixels(points[0][0])}
        cy={unitsToPixels(points[0][1])}
        r={unitsToPixels(jointMergeRadius)}
        fill="none"
        stroke="#e91e63"
        strokeWidth={2}
        strokeDasharray="4,4"
        opacity={0.6}
      />
    ) : null

    // Draw points
    const pointMarkers = points.map((point, i) => (
      <circle
        key={`path-point-${i}`}
        cx={unitsToPixels(point[0])}
        cy={unitsToPixels(point[1])}
        r={i === 0 ? 8 : 5}
        fill={i === 0 ? '#e91e63' : '#f48fb1'}
        stroke="#fff"
        strokeWidth={2}
        opacity={0.9}
      />
    ))

    return (
      <g>
        {closingLine}
        {lines}
        {snapCircle}
        {pointMarkers}
      </g>
    )
  }

  // Render completed target paths (for trajectory optimization)
  // All target paths are rendered as closed curves (cyclic trajectories)
  const renderTargetPaths = () => {
    if (targetPaths.length === 0) return null

    return targetPaths.map(path => {
      const isSelected = selectedPathId === path.id
      const points = path.points

      if (points.length < 2) return null

      // Create path data for closed curve (Z closes the path back to start)
      const pathData = points.map((p, i) =>
        `${i === 0 ? 'M' : 'L'} ${unitsToPixels(p[0])} ${unitsToPixels(p[1])}`
      ).join(' ') + ' Z'  // Close the path

      return (
        <g key={path.id}>
          {/* Path line - closed shape */}
          <path
            d={pathData}
            fill="none"
            stroke={path.color}
            strokeWidth={isSelected ? 4 : 3}
            strokeDasharray="10,5"
            opacity={isSelected ? 1 : 0.7}
            style={{ cursor: 'pointer' }}
            onClick={(e) => {
              e.stopPropagation()
              setSelectedPathId(isSelected ? null : path.id)
            }}
          />
          {/* Path points */}
          {points.map((point, i) => (
            <circle
              key={`${path.id}-point-${i}`}
              cx={unitsToPixels(point[0])}
              cy={unitsToPixels(point[1])}
              r={isSelected ? 5 : 4}
              fill={path.color}
              stroke="#fff"
              strokeWidth={1.5}
              opacity={isSelected ? 1 : 0.7}
              style={{ pointerEvents: 'none' }}
            />
          ))}
          {/* Start point indicator (larger) */}
          <circle
            cx={unitsToPixels(points[0][0])}
            cy={unitsToPixels(points[0][1])}
            r={isSelected ? 7 : 6}
            fill={path.color}
            stroke="#fff"
            strokeWidth={2}
            opacity={isSelected ? 1 : 0.8}
            style={{ pointerEvents: 'none' }}
          />
          {/* Path label */}
          {isSelected && points.length > 0 && (
            <text
              x={unitsToPixels(points[0][0])}
              y={unitsToPixels(points[0][1]) - 14}
              textAnchor="middle"
              fontSize="11"
              fontWeight="bold"
              fill={path.color}
            >
              {path.name} (closed)
            </text>
          )}
        </g>
      )
    })
  }

  // Render measurement markers (X marks that fade)
  const renderMeasurementMarkers = () => {
    const now = Date.now()

    return measurementMarkers.map((marker) => {
      const age = now - marker.timestamp
      const opacity = Math.max(0, 1 - age / 3000) // Fade over 3 seconds
      const size = 8
      const x = unitsToPixels(marker.point[0])
      const y = unitsToPixels(marker.point[1])

      return (
        <g key={marker.id} opacity={opacity}>
          {/* X mark */}
          <line
            x1={x - size}
            y1={y - size}
            x2={x + size}
            y2={y + size}
            stroke="#f44336"
            strokeWidth={3}
            strokeLinecap="round"
          />
          <line
            x1={x + size}
            y1={y - size}
            x2={x - size}
            y2={y + size}
            stroke="#f44336"
            strokeWidth={3}
            strokeLinecap="round"
          />
          {/* Coordinate label */}
          <text
            x={x}
            y={y - 14}
            textAnchor="middle"
            fontSize="10"
            fill="#f44336"
            fontWeight="500"
          >
            ({marker.point[0].toFixed(1)}, {marker.point[1].toFixed(1)})
          </text>
        </g>
      )
    })
  }

  // Render measurement line (if measuring)
  const renderMeasurementLine = () => {
    if (!measureState.isMeasuring || !measureState.startPoint) return null

    const startX = unitsToPixels(measureState.startPoint[0])
    const startY = unitsToPixels(measureState.startPoint[1])

    return (
      <g>
        {/* Start point indicator */}
        <circle
          cx={startX}
          cy={startY}
          r={6}
          fill="#f44336"
          stroke="#fff"
          strokeWidth={2}
        />
      </g>
    )
  }

  // Toolbar toggle handler
  const handleToggleToolbar = useCallback((id: string) => {
    setOpenToolbars(prev => {
      const newSet = new Set(prev)
      if (newSet.has(id)) {
        newSet.delete(id)
      } else {
        newSet.add(id)
      }
      return newSet
    })
  }, [])

  // Toolbar position change handler
  const handleToolbarPositionChange = useCallback((id: string, position: ToolbarPosition) => {
    setToolbarPositions(prev => ({ ...prev, [id]: position }))
  }, [])

  // Get toolbar position (use saved or default)
  // Negative x values mean "offset from right edge of canvas"
  const getToolbarPosition = (id: string): ToolbarPosition => {
    if (toolbarPositions[id]) return toolbarPositions[id]
    const config = TOOLBAR_CONFIGS.find(c => c.id === id)
    const defaultPos = config?.defaultPosition || { x: 100, y: 100 }

    // Convert negative x to position from right edge
    if (defaultPos.x < 0) {
      return {
        x: canvasDimensions.width + defaultPos.x,
        y: defaultPos.y
      }
    }
    return defaultPos
  }

  // Get toolbar dimensions based on type
  const getToolbarDimensions = (id: string): { minWidth: number; maxHeight: number } => {
    switch (id) {
      case 'tools':
        // Tools should NEVER scroll - large maxHeight to fit all content
        return { minWidth: 220, maxHeight: 600 }
      case 'more':
        return { minWidth: 180, maxHeight: 350 }  // Simplified, shorter now (optimization moved out)
      case 'optimize':
        return { minWidth: 320, maxHeight: 700 }  // Wide for hyperparameters, tall for all controls
      case 'links':
        return { minWidth: 200, maxHeight: 480 }  // 1.5x taller for links list
      case 'nodes':
        return { minWidth: 200, maxHeight: 320 }  // Taller for nodes list
      case 'settings':
        return { minWidth: 280, maxHeight: 900 }  // Wide enough for form fields, tall for all settings sections
      default:
        return { minWidth: 200, maxHeight: 400 }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  // TOOLBAR CONTENT COMPONENTS
  // ═══════════════════════════════════════════════════════════════════════════════

  // Tools toolbar content - fixed width to prevent expansion
  const TOOL_BUTTON_SIZE = 48
  const TOOLS_GRID_GAP = 6
  const TOOLS_PADDING = 12
  const TOOLS_BOX_WIDTH = (TOOL_BUTTON_SIZE * 3) + (TOOLS_GRID_GAP * 2) + (TOOLS_PADDING * 2)

  const ToolsContent = () => (
    <Box sx={{ p: `${TOOLS_PADDING}px`, width: TOOLS_BOX_WIDTH, boxSizing: 'border-box' }}>
      <Box sx={{
        display: 'grid',
        gridTemplateColumns: `repeat(3, ${TOOL_BUTTON_SIZE}px)`,
        gap: `${TOOLS_GRID_GAP}px`,
        justifyContent: 'center'
      }}>
        {TOOLS.map(tool => {
          const isActive = toolMode === tool.id
          const isHovered = hoveredTool === tool.id
          const isDelete = tool.id === 'delete'

          return (
            <Tooltip
              key={tool.id}
              title={
                <Box sx={{ p: 0.5 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                    {tool.label} {tool.shortcut && `(${tool.shortcut})`}
                  </Typography>
                  <Typography variant="caption" sx={{ display: 'block', mt: 0.5, opacity: 0.9 }}>
                    {tool.description}
                  </Typography>
                </Box>
              }
              placement="bottom"
              arrow
            >
              <IconButton
                onClick={() => {
                  if (linkCreationState.isDrawing && tool.id !== 'draw_link') {
                    setLinkCreationState(initialLinkCreationState)
                    setPreviewLine(null)
                  }
                  setToolMode(tool.id)
                }}
                onMouseEnter={() => setHoveredTool(tool.id)}
                onMouseLeave={() => setHoveredTool(null)}
                sx={{
                  width: TOOL_BUTTON_SIZE,
                  height: TOOL_BUTTON_SIZE,
                  minWidth: TOOL_BUTTON_SIZE,
                  maxWidth: TOOL_BUTTON_SIZE,
                  borderRadius: 2,
                  fontSize: isDelete ? '1.2rem' : '1.5rem',
                  overflow: 'hidden',
                  backgroundColor: isActive
                    ? (isDelete ? '#d32f2f' : 'primary.main')
                    : (isHovered ? 'rgba(0,0,0,0.04)' : 'transparent'),
                  color: isActive ? 'white' : (isDelete && isHovered ? '#d32f2f' : 'text.primary'),
                  border: isActive ? 'none' : '1px solid transparent',
                  transition: 'all 0.15s ease',
                  '&:hover': {
                    backgroundColor: isActive
                      ? (isDelete ? '#b71c1c' : 'primary.dark')
                      : (isDelete ? 'rgba(211, 47, 47, 0.08)' : 'rgba(0,0,0,0.06)'),
                    border: isActive ? 'none' : '1px solid rgba(0,0,0,0.12)'
                  }
                }}
              >
                <span style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  width: '100%',
                  height: '100%',
                  lineHeight: 1
                }}>
                {tool.icon}
                </span>
              </IconButton>
            </Tooltip>
          )
        })}
      </Box>

      <Divider sx={{ my: 1 }} />

      <Box sx={{ px: 0.5 }}>
        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
          {TOOLS.find(t => t.id === toolMode)?.label}
        </Typography>
        <Typography variant="caption" display="block" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
          {TOOLS.find(t => t.id === toolMode)?.description}
        </Typography>
      </Box>
    </Box>
  )

  // Update link property
  const updateLinkProperty = useCallback((linkName: string, property: string, value: string | string[] | boolean) => {
    setPylinkDoc(prev => ({
      ...prev,
      meta: {
        ...prev.meta,
        links: {
          ...prev.meta.links,
          [linkName]: {
            ...prev.meta.links[linkName],
            [property]: value
          }
        }
      }
    }))
  }, [])

  // Rename a link
  const renameLink = useCallback((oldName: string, newName: string) => {
    if (oldName === newName || !newName.trim()) return
    if (pylinkDoc.meta.links[newName]) {
      showStatus(`Link "${newName}" already exists`, 'error', 2000)
      return
    }
    setPylinkDoc(prev => {
      const newLinks = { ...prev.meta.links }
      newLinks[newName] = newLinks[oldName]
      delete newLinks[oldName]
      return { ...prev, meta: { ...prev.meta, links: newLinks } }
    })
    // Update the modal data with new name if modal is open
    if (editingLinkData && editingLinkData.name === oldName) {
      setEditingLinkData(prev => prev ? { ...prev, name: newName } : null)
    }
    showStatus(`Renamed to ${newName}`, 'success', 1500)
  }, [pylinkDoc.meta.links, showStatus, editingLinkData])

  // Track click timing for distinguishing single vs double click
  const linkClickTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Links toolbar content - double-click opens edit modal
  const LinksContent = () => (
    <Box sx={{ overflow: 'auto', maxHeight: 350 }}>
      {Object.entries(pylinkDoc.meta.links).length === 0 ? (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">No links yet</Typography>
          <Typography variant="caption" display="block" color="text.disabled">Use Draw Link tool (L)</Typography>
        </Box>
      ) : (
        Object.entries(pylinkDoc.meta.links).map(([linkName, linkMeta], index) => {
          const pos0 = getJointPosition(linkMeta.connects[0])
          const pos1 = getJointPosition(linkMeta.connects[1])
          const length = pos0 && pos1 ? calculateDistance(pos0, pos1) : null
          const isSelected = selectedLinks.includes(linkName)
          const isHovered = hoveredLink === linkName

          const handleClick = (e: React.MouseEvent) => {
            // Clear any pending single-click action
            if (linkClickTimeoutRef.current) {
              clearTimeout(linkClickTimeoutRef.current)
              linkClickTimeoutRef.current = null
            }

            if (e.detail === 2) {
              // Double click - open modal
              openLinkEditModal(linkName)
            } else {
              // Single click - delay selection to allow for double-click
              linkClickTimeoutRef.current = setTimeout(() => {
                setSelectedLinks([linkName])
                setSelectedJoints([])
                linkClickTimeoutRef.current = null
              }, 200)
            }
          }

          return (
            <Box
              key={linkName}
              onMouseEnter={() => setHoveredLink(linkName)}
              onMouseLeave={() => setHoveredLink(null)}
              onClick={handleClick}
              sx={{
                py: 0.75, px: 1.5, cursor: 'pointer',
                backgroundColor: isSelected ? `${selectionColor}14` : (isHovered ? `${selectionColor}1f` : 'transparent'),
                borderLeft: `3px solid ${linkMeta.color || getDefaultColor(index)}`,
                '&:hover': { backgroundColor: `${selectionColor}1f` },
                transition: 'background-color 0.15s ease'
              }}
            >
              <Typography sx={{ fontSize: '0.75rem', fontWeight: isHovered ? 600 : 500 }}>{linkName}</Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                {linkMeta.connects.join(' → ')}{length ? ` • ${length.toFixed(1)}` : ''}
              </Typography>
            </Box>
          )
        })
      )}
    </Box>
  )

  // Valid pylinkage joint types
  const JOINT_TYPES = ['Static', 'Crank', 'Revolute'] as const

  // Update joint property - immediately updates the joint type in pylinkDoc
  // IMPORTANT: For non-Static types, we store the UI position in meta.joints
  // This is the SINGLE SOURCE OF TRUTH for visual position
  const updateJointProperty = useCallback((jointName: string, property: string, value: string) => {
    setPylinkDoc(prev => {
      const jointIndex = prev.pylinkage.joints.findIndex(j => j.name === jointName)
      if (jointIndex === -1) return prev

      const newJoints = [...prev.pylinkage.joints]
      const currentJoint = newJoints[jointIndex]

      if (property === 'type') {
        // Get current position BEFORE changing type (single source of truth)
        const pos = getJointPosition(jointName)
        if (!pos) return prev

        const newType = value as 'Static' | 'Crank' | 'Revolute'

        // Prepare updated meta.joints with position preserved
        const newMetaJoints = { ...prev.meta.joints }
        const currentMeta = newMetaJoints[jointName] || { color: '', zlevel: 0 }

        if (newType === 'Static') {
          // Convert to Static: store position in pylinkage data, remove from meta
          newJoints[jointIndex] = {
            type: 'Static',
            name: jointName,
            x: pos[0],
            y: pos[1]
          } as StaticJoint
          // Remove UI position from meta (Static uses pylinkage x,y)
          newMetaJoints[jointName] = { ...currentMeta, x: undefined, y: undefined }
        } else if (newType === 'Crank') {
          // Convert to Crank: store UI position in meta
          const existingParent = currentJoint.type === 'Crank'
            ? currentJoint.joint0.ref
            : currentJoint.type === 'Revolute'
              ? currentJoint.joint0.ref
              : prev.pylinkage.joints.find(j => j.type === 'Static' && j.name !== jointName)?.name

          if (!existingParent) {
            showStatus('Need a static joint to reference for Crank type', 'error', 2000)
            return prev
          }

          const parentPos = getJointPosition(existingParent)
          const distance = parentPos ? calculateDistance(parentPos, pos) : 10
          const angle = parentPos ? Math.atan2(pos[1] - parentPos[1], pos[0] - parentPos[0]) : 0

          newJoints[jointIndex] = {
            type: 'Crank',
            name: jointName,
            joint0: { ref: existingParent },
            distance: distance,
            angle: angle
          } as CrankJoint

          // Store UI position in meta (single source of truth)
          newMetaJoints[jointName] = { ...currentMeta, x: pos[0], y: pos[1], show_path: true }
        } else if (newType === 'Revolute') {
          // Convert to Revolute: store UI position in meta
          // IMPORTANT: Look at visual links to determine actual parent joints!
          const connectedJoints = getConnectedJointsFromLinks(jointName, prev.meta.links)

          let existingParent0: string | undefined
          let existingParent1: string | undefined

          if (connectedJoints.length >= 2) {
            // Use the joints that are visually connected via links
            existingParent0 = connectedJoints[0]
            existingParent1 = connectedJoints[1]
          } else if (currentJoint.type === 'Revolute') {
            // Keep existing references if already Revolute
            existingParent0 = currentJoint.joint0.ref
            existingParent1 = currentJoint.joint1.ref
          } else {
            // Fallback: use any two joints (not ideal, but prevents crash)
            existingParent0 = currentJoint.type === 'Crank'
              ? currentJoint.joint0.ref
              : prev.pylinkage.joints.find(j => j.name !== jointName)?.name
            existingParent1 = prev.pylinkage.joints.find(
              j => j.name !== jointName && j.name !== existingParent0
            )?.name
          }

          if (!existingParent0 || !existingParent1) {
            showStatus('Need two joints to reference for Revolute type (draw links first)', 'error', 2000)
            return prev
          }

          const parent0Pos = getJointPosition(existingParent0)
          const parent1Pos = getJointPosition(existingParent1)
          const distance0 = parent0Pos ? calculateDistance(parent0Pos, pos) : 10
          const distance1 = parent1Pos ? calculateDistance(parent1Pos, pos) : 10

          newJoints[jointIndex] = {
            type: 'Revolute',
            name: jointName,
            joint0: { ref: existingParent0 },
            joint1: { ref: existingParent1 },
            distance0: distance0,
            distance1: distance1
          } as RevoluteJoint

          // Store UI position in meta (single source of truth)
          newMetaJoints[jointName] = { ...currentMeta, x: pos[0], y: pos[1], show_path: true }
        }

        showStatus(`Changed ${jointName} to ${newType}`, 'success', 1500)

        // Clear trajectory (will trigger auto-simulation via effect)
        clearTrajectory()

        return {
          ...prev,
          pylinkage: { ...prev.pylinkage, joints: newJoints },
          meta: { ...prev.meta, joints: newMetaJoints }
        }
      }

      return prev
    })
    // Trigger auto-simulation after state update
    triggerMechanismChange()
  }, [getJointPosition, showStatus, triggerMechanismChange])

  // Rename a joint
  const renameJoint = useCallback((oldName: string, newName: string) => {
    if (oldName === newName || !newName.trim()) return
    if (pylinkDoc.pylinkage.joints.some(j => j.name === newName)) {
      showStatus(`Joint "${newName}" already exists`, 'error', 2000)
      return
    }
    setPylinkDoc(prev => {
      // Update joint name
      const newJoints = prev.pylinkage.joints.map(j =>
        j.name === oldName ? { ...j, name: newName } : j
      )
      // Update solve_order
      const newSolveOrder = prev.pylinkage.solve_order.map(n => n === oldName ? newName : n)
      // Update meta joints
      const newMetaJoints = { ...prev.meta.joints }
      if (newMetaJoints[oldName]) {
        newMetaJoints[newName] = newMetaJoints[oldName]
        delete newMetaJoints[oldName]
      }
      // Update links that reference this joint
      const newLinks: Record<string, { color: string; connects: string[] }> = {}
      Object.entries(prev.meta.links).forEach(([linkName, link]) => {
        newLinks[linkName] = {
          ...link,
          connects: link.connects.map(c => c === oldName ? newName : c)
        }
      })
      // Update joint references in other joints (Crank, Revolute)
      const updatedJoints = newJoints.map(j => {
        if (j.type === 'Crank' && j.joint0?.ref === oldName) {
          return { ...j, joint0: { ref: newName } }
        }
        if (j.type === 'Revolute') {
          const updated = { ...j }
          if (j.joint0?.ref === oldName) updated.joint0 = { ref: newName }
          if (j.joint1?.ref === oldName) updated.joint1 = { ref: newName }
          return updated
        }
        return j
      })

      return {
        ...prev,
        pylinkage: { ...prev.pylinkage, joints: updatedJoints, solve_order: newSolveOrder },
        meta: { ...prev.meta, joints: newMetaJoints, links: newLinks }
      }
    })
    // Update the modal data with new name if modal is open
    if (editingJointData && editingJointData.name === oldName) {
      setEditingJointData(prev => prev ? { ...prev, name: newName } : null)
    }
    showStatus(`Renamed to ${newName}`, 'success', 1500)
  }, [pylinkDoc.pylinkage.joints, showStatus, editingJointData])

  // Track click timing for distinguishing single vs double click
  const jointClickTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Nodes toolbar content - double-click opens edit modal
  const NodesContent = () => (
    <Box sx={{ overflow: 'auto', maxHeight: 350 }}>
      {pylinkDoc.pylinkage.joints.length === 0 ? (
        <Box sx={{ p: 2, textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">No joints yet</Typography>
          <Typography variant="caption" display="block" color="text.disabled">Use Draw Link tool (L)</Typography>
        </Box>
      ) : (
        pylinkDoc.pylinkage.joints.map((joint) => {
          const pos = getJointPosition(joint.name)
          const isSelected = selectedJoints.includes(joint.name)
          const isHovered = hoveredJoint === joint.name
          const typeColor = joint.type === 'Static' ? jointColors.static : joint.type === 'Crank' ? jointColors.crank : jointColors.pivot

          const handleClick = (e: React.MouseEvent) => {
            // Clear any pending single-click action
            if (jointClickTimeoutRef.current) {
              clearTimeout(jointClickTimeoutRef.current)
              jointClickTimeoutRef.current = null
            }

            if (e.detail === 2) {
              // Double click - open modal
              openJointEditModal(joint.name)
            } else {
              // Single click - delay selection to allow for double-click
              jointClickTimeoutRef.current = setTimeout(() => {
                setSelectedJoints([joint.name])
                setSelectedLinks([])
                jointClickTimeoutRef.current = null
              }, 200)
            }
          }

          return (
            <Box
              key={joint.name}
              onMouseEnter={() => setHoveredJoint(joint.name)}
              onMouseLeave={() => setHoveredJoint(null)}
              onClick={handleClick}
              sx={{
                py: 0.75, px: 1.5, cursor: 'pointer',
                backgroundColor: isSelected ? `${selectionColor}14` : (isHovered ? `${selectionColor}1f` : 'transparent'),
                borderLeft: `3px solid ${typeColor}`,
                '&:hover': { backgroundColor: `${selectionColor}1f` },
                transition: 'background-color 0.15s ease'
              }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                <Typography sx={{ fontSize: '0.75rem', fontWeight: isHovered ? 600 : 500 }}>{joint.name}</Typography>
                <Chip
                  label={joint.type}
                  size="small"
                  sx={{
                    height: 14, fontSize: '0.5rem',
                    backgroundColor: joint.type === 'Static' ? '#ffebee' : joint.type === 'Crank' ? '#fff3e0' : '#e3f2fd'
                  }}
                />
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
                {pos ? `(${pos[0].toFixed(1)}, ${pos[1].toFixed(1)})` : ''}
              </Typography>
            </Box>
          )
        })
      )}
    </Box>
  )

  // More toolbar content
  const MoreContent = () => {
    const hasCrank = canSimulate(pylinkDoc.pylinkage.joints)
    const canAnimate = trajectoryData !== null && trajectoryData.nSteps > 0 && stretchingLinks.length === 0
    const hasStretchingLinks = stretchingLinks.length > 0

    return (
      <Box sx={{ p: 1.5 }}>
        {/* ═══════════════════════════════════════════════════════════════════════
            ANIMATION - Plays through simulation frames to animate linkage motion
            ═══════════════════════════════════════════════════════════════════════ */}
        <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
          Animation
        </Typography>
        <Box sx={{ display: 'flex', gap: 0.5, mt: 1, mb: 1 }}>
          <Tooltip
            title={
              <Box>
                <Typography variant="body2">
                  {animationState.isAnimating ? 'Pause animation' : 'Play animation'}
                </Typography>
                <Typography variant="caption" sx={{ opacity: 0.8 }}>
                  Shortcut: Spacebar
                </Typography>
              </Box>
            }
            enterDelay={400}
            leaveDelay={100}
            placement="top"
            arrow
          >
            <span style={{ flex: 1 }}>
              <Button
                variant="contained"
                fullWidth
                size="small"
                onMouseDown={(e) => {
                  // Use onMouseDown for more responsive click during animation
                  e.preventDefault()
                  if (hasStretchingLinks) {
                    showStatus(
                      `Cannot animate: ${stretchingLinks.join(', ')} would stretch. Fix kinematic constraints first.`,
                      'error',
                      3000
                    )
                    return
                  }
                  if (animationState.isAnimating) {
                    pauseAnimation()
                  } else {
                    // If no trajectory data, run simulation first then play
                    if (!canAnimate) {
                      runSimulation().then(() => {
                        setTimeout(() => playAnimation(), 100)
                      })
                    } else {
                      playAnimation()
                    }
                  }
                }}
                disabled={!hasCrank || isSimulating || hasStretchingLinks}
                sx={{
                  textTransform: 'none', fontSize: '0.75rem',
                  backgroundColor: animationState.isAnimating ? '#ff9800' : '#4caf50',
                  '&:hover': { backgroundColor: animationState.isAnimating ? '#f57c00' : '#388e3c' },
                  '&.Mui-disabled': { backgroundColor: '#e0e0e0' },
                  pointerEvents: 'auto'  // Ensure clicks register during animation
                }}
              >
                {animationState.isAnimating ? '⏸ Pause' : '▶ Play'}
              </Button>
            </span>
          </Tooltip>
          <Tooltip
            title="Reset: Returns the mechanism to its starting position (frame 0)"
            enterDelay={400}
            leaveDelay={100}
            placement="top"
            arrow
          >
            <span>
              <Button
                variant="outlined"
                size="small"
                onClick={() => {
                  stopAnimation()
                  setAnimatedPositions(null)
                }}
                disabled={!canAnimate && animationState.currentFrame === 0}
                sx={{
                  textTransform: 'none', fontSize: '0.75rem', minWidth: 40,
                  borderColor: '#666', color: '#666'
                }}
              >
                ↺
              </Button>
            </span>
          </Tooltip>
        </Box>

        {/* Animation info & speed control */}
        {canAnimate && (
          <Box sx={{ mb: 1.5 }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
              Frame: {animationState.currentFrame + 1} / {animationState.totalFrames}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5 }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.65rem' }}>
                Speed:
              </Typography>
              {[0.5, 1, 2].map(speed => (
                <Box
                  key={speed}
                  onClick={() => setPlaybackSpeed(speed)}
                  sx={{
                    px: 0.75, py: 0.25, borderRadius: 1,
                    fontSize: '0.65rem', cursor: 'pointer',
                    bgcolor: animationState.playbackSpeed === speed ? 'primary.main' : 'grey.100',
                    color: animationState.playbackSpeed === speed ? '#fff' : 'text.secondary',
                    '&:hover': { bgcolor: animationState.playbackSpeed === speed ? 'primary.dark' : 'grey.200' }
                  }}
                >
                  {speed}x
                </Box>
              ))}
            </Box>
          </Box>
        )}

        <Divider sx={{ my: 1 }} />

        {/* ═══════════════════════════════════════════════════════════════════════
            SIMULATION - Computes and displays trajectory dots
            ═══════════════════════════════════════════════════════════════════════ */}
        <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
          Trajectory Simulation
        </Typography>
        <Box sx={{ display: 'flex', gap: 0.5, mt: 1, mb: 1 }}>
          <Tooltip
            title={
              <Box sx={{ p: 0.5, maxWidth: 280 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600, fontSize: '0.75rem' }}>
                  {autoSimulateEnabled ? 'Disable Continuous Simulation' : 'Enable Continuous Simulation'}
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', mt: 0.5, fontSize: '0.65rem' }}>
                  When enabled, the trajectory is automatically recomputed whenever you modify the mechanism.
                  Delay: {autoSimulateDelayMs}ms (configurable in Settings).
                </Typography>
              </Box>
            }
            enterDelay={500}
            leaveDelay={100}
            placement="top"
            arrow
          >
            <span style={{ flex: 1 }}>
              <Button
                variant="contained"
                fullWidth
                size="small"
                onClick={() => {
                  if (autoSimulateEnabled) {
                    setAutoSimulateEnabled(false)
                  } else {
                    setAutoSimulateEnabled(true)
                    triggerMechanismChange()  // Trigger initial simulation
                  }
                }}
                disabled={isSimulating || !hasCrank}
                sx={{
                  textTransform: 'none', fontSize: '0.7rem',
                  backgroundColor: autoSimulateEnabled ? '#2196f3' : '#666',
                  '&:hover': { backgroundColor: autoSimulateEnabled ? '#1976d2' : '#555' }
                }}
              >
                {autoSimulateEnabled ? '◉ Continuous Simulation' : '○ Continuous Simulation'}
              </Button>
            </span>
          </Tooltip>
        </Box>

        {/* Show/Hide trajectory toggle when we have data */}
        {trajectoryData && (
          <Box sx={{ mb: 1 }}>
            <Button
              variant={showTrajectory ? 'contained' : 'outlined'}
              fullWidth
              size="small"
              onClick={() => setShowTrajectory(!showTrajectory)}
              sx={{
                textTransform: 'none', fontSize: '0.7rem',
                backgroundColor: showTrajectory ? '#9c27b0' : 'transparent',
                borderColor: '#9c27b0', color: showTrajectory ? '#fff' : '#9c27b0',
                '&:hover': { backgroundColor: showTrajectory ? '#7b1fa2' : 'rgba(156, 39, 176, 0.1)' }
              }}
            >
              {showTrajectory ? 'Hide Pathss' : 'Show All Paths'}
            </Button>
          </Box>
        )}

        <Divider sx={{ my: 1 }} />

        {/* ═══════════════════════════════════════════════════════════════════════
            DEMOS
            ═══════════════════════════════════════════════════════════════════════ */}
        <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
          Demos
        </Typography>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={loadDemo4Bar}
          sx={{ mt: 1, mb: 1.5, textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
        >
          ◇ Four Bar Demo
        </Button>

        <Divider sx={{ my: 1 }} />

        {/* ═══════════════════════════════════════════════════════════════════════
            FILE OPERATIONS
            ═══════════════════════════════════════════════════════════════════════ */}
        <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
          File Operations
        </Typography>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={loadPylinkGraph}
          sx={{ mt: 1, mb: 1, textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
        >
          ↑ Load
        </Button>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={savePylinkGraph}
          sx={{ textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
        >
          ↓ Save
        </Button>

        <Divider sx={{ my: 1 }} />

        {/* ═══════════════════════════════════════════════════════════════════════
            VALIDATION
            ═══════════════════════════════════════════════════════════════════════ */}
        <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
          Validation
        </Typography>
        <Tooltip
          title={
            <Box sx={{ p: 0.5, maxWidth: 240 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, fontSize: '0.75rem' }}>
                Validate Mechanism
              </Typography>
              <Typography variant="caption" sx={{ display: 'block', mt: 0.5, fontSize: '0.65rem' }}>
                Checks if the mechanism can be simulated. Requires links, a Crank driver, and Static ground.
              </Typography>
            </Box>
          }
          enterDelay={500}
          leaveDelay={100}
          placement="top"
          arrow
        >
          <span style={{ display: 'block', marginTop: 8 }}>
            <Button
              variant="outlined"
              fullWidth
              size="small"
              onClick={validateMechanism}
              sx={{
                textTransform: 'none',
                justifyContent: 'flex-start',
                fontSize: '0.75rem',
                borderColor: '#1976d2',
                color: '#1976d2',
                '&:hover': {
                  backgroundColor: 'rgba(25, 118, 210, 0.08)',
                  borderColor: '#1565c0'
                }
              }}
            >
              ✓ Validate
            </Button>
          </span>
        </Tooltip>
      </Box>
    )
  }

  // Run trajectory optimization
  // Extract current dimensions from pylink document (for showing before/after)
  const extractCurrentDimensions = (doc: PylinkDocument): Record<string, number> => {
    const dims: Record<string, number> = {}
    for (const joint of doc.pylinkage.joints) {
      if (joint.type === 'Crank' && joint.distance !== undefined) {
        dims[`${joint.name}_distance`] = joint.distance
      } else if (joint.type === 'Revolute') {
        if (joint.distance0 !== undefined) dims[`${joint.name}_distance0`] = joint.distance0
        if (joint.distance1 !== undefined) dims[`${joint.name}_distance1`] = joint.distance1
      }
    }
    return dims
  }

  // Revert to pre-optimization state
  const revertOptimization = () => {
    if (preOptimizationDoc) {
      setPylinkDoc(preOptimizationDoc)
      setPreOptimizationDoc(null)
      setOptimizationResult(null)
      triggerMechanismChange()
      showStatus('Reverted to pre-optimization state', 'info', 2000)
    }
  }

  // Preprocess trajectory (smooth and/or resample)
  const preprocessTrajectory = async () => {
    const selectedPath = targetPaths.find(p => p.id === selectedPathId)
    if (!selectedPath || selectedPath.points.length < 3) {
      showStatus('Select a path with at least 3 points', 'warning', 2000)
      return
    }

    try {
      setIsPreprocessing(true)
      setPreprocessResult(null)
      showStatus('Preprocessing trajectory...', 'action')

      const response = await fetch('/api/prepare-trajectory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trajectory: selectedPath.points,
          target_n_steps: prepTargetNSteps,
          smooth: prepEnableSmooth,
          smooth_window: prepSmoothWindow,
          smooth_polyorder: prepSmoothPolyorder,
          smooth_method: prepSmoothMethod,
          resample: prepEnableResample,
          resample_method: prepResampleMethod,
          closed: true  // All target paths are treated as closed/cyclic
        })
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success') {
        // Update the path with preprocessed points
        const newPoints: [number, number][] = result.trajectory.map((p: number[]) => [p[0], p[1]] as [number, number])

        setTargetPaths(prev => prev.map(p =>
          p.id === selectedPathId
            ? { ...p, points: newPoints, name: `${p.name} (processed)` }
            : p
        ))

        setPreprocessResult({
          originalPoints: result.original_points,
          outputPoints: result.output_points,
          analysis: result.analysis
        })

        showStatus(
          `Preprocessed: ${result.original_points} → ${result.output_points} points`,
          'success',
          3000
        )
      } else {
        showStatus(result.message || 'Preprocessing failed', 'error', 3000)
      }
    } catch (error) {
      showStatus(`Preprocessing error: ${error}`, 'error', 3000)
    } finally {
      setIsPreprocessing(false)
    }
  }

  const runOptimization = async () => {
    const selectedPath = targetPaths.find(p => p.id === selectedPathId)
    if (!selectedPath || !selectedPath.targetJoint) {
      showStatus('Select a path and target joint first', 'warning', 2000)
      return
    }

    // Warn if target path points don't match simulation steps
    if (selectedPath.points.length !== simulationSteps) {
      console.warn(`⚠️ Path has ${selectedPath.points.length} points but simulation uses ${simulationSteps} steps. Consider preprocessing the path.`)
    }

    try {
      setIsOptimizing(true)
      setOptimizationResult(null)
      showStatus(`Running ${optMethod.toUpperCase()} optimization (N=${simulationSteps})...`, 'action')

      // Save current state before optimization (deep copy)
      const savedDoc = JSON.parse(JSON.stringify(pylinkDoc)) as PylinkDocument
      setPreOptimizationDoc(savedDoc)

      // Extract original dimensions for comparison
      const originalDims = extractCurrentDimensions(pylinkDoc)

      // Build optimization options based on method
      const optimizationOptions: Record<string, unknown> = {
        method: optMethod,
        verbose: optVerbose,
        bounds_factor: optBoundsFactor,
        min_length: optMinLength
      }

      // PSO-specific options
      if (optMethod === 'pso' || optMethod === 'pylinkage') {
        optimizationOptions.n_particles = optNParticles
        optimizationOptions.iterations = optIterations
      }

      // SciPy-specific options
      if (optMethod === 'scipy' || optMethod === 'powell' || optMethod === 'nelder-mead') {
        optimizationOptions.max_iterations = optMaxIterations
        optimizationOptions.tolerance = optTolerance
      }

      const response = await fetch('/api/optimize-trajectory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pylink_data: {
            ...pylinkDoc,
            n_steps: simulationSteps  // Use simulation steps for consistency
          },
          target_path: {
            joint_name: selectedPath.targetJoint,
            positions: selectedPath.points
          },
          optimization_options: optimizationOptions
        })
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success' && result.result) {
        const optResult = result.result

        // Apply the optimized pylink data (backend now returns correct meta.joints positions)
        if (optResult.optimized_pylink_data) {
          // Backend has already updated meta.joints positions, just apply directly
          setPylinkDoc(optResult.optimized_pylink_data as PylinkDocument)
          triggerMechanismChange()
        }

        const improvement = optResult.initial_error > 0
          ? ((1 - optResult.final_error / optResult.initial_error) * 100)
          : 0

        setOptimizationResult({
          success: optResult.success,
          initialError: optResult.initial_error,
          finalError: optResult.final_error,
          message: result.message,
          iterations: optResult.iterations,
          executionTimeMs: result.execution_time_ms,
          optimizedDimensions: optResult.optimized_dimensions,
          originalDimensions: originalDims
        })

        if (optResult.success) {
          showStatus(`Optimization complete: ${improvement.toFixed(1)}% improvement`, 'success', 4000)
        } else {
          showStatus(`Optimization finished with limited improvement: ${improvement.toFixed(1)}%`, 'warning', 4000)
        }
      } else {
        // Optimization failed - clear the saved state
        setPreOptimizationDoc(null)
        setOptimizationResult({
          success: false,
          initialError: 0,
          finalError: 0,
          message: result.message || 'Optimization failed'
        })
        showStatus(result.message || 'Optimization failed', 'error', 3000)
      }
    } catch (error) {
      // Error - clear the saved state
      setPreOptimizationDoc(null)
      setOptimizationResult({
        success: false,
        initialError: 0,
        finalError: 0,
        message: `Error: ${error}`
      })
      showStatus(`Optimization error: ${error}`, 'error', 3000)
    } finally {
      setIsOptimizing(false)
    }
  }

  // Settings toolbar content
  const SettingsContent = () => (
    <Box sx={{ p: 1.5, minWidth: 240 }}>
      {/* ═══════════════════════════════════════════════════════════════════════
          APPEARANCE SETTINGS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Appearance
      </Typography>

      {/* Dark Mode Toggle */}
      <FormControlLabel
        control={
          <Switch
            checked={darkMode}
            onChange={(e) => setDarkMode(e.target.checked)}
            size="small"
          />
        }
        label={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <span>{darkMode ? '🌙' : '☀️'}</span>
            <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
              Dark Mode
            </Typography>
          </Box>
        }
        sx={{ mt: 1, mb: 0.5, ml: 0 }}
      />

      {/* Show Grid Toggle */}
      <FormControlLabel
        control={
          <Switch
            checked={showGrid}
            onChange={(e) => setShowGrid(e.target.checked)}
            size="small"
          />
        }
        label={<Typography variant="body2" sx={{ fontSize: '0.8rem' }}>Show Grid</Typography>}
        sx={{ mb: 0.5, ml: 0 }}
      />

      {/* Show Joint Labels Toggle */}
      <FormControlLabel
        control={
          <Switch
            checked={showJointLabels}
            onChange={(e) => setShowJointLabels(e.target.checked)}
            size="small"
          />
        }
        label={<Typography variant="body2" sx={{ fontSize: '0.8rem' }}>Joint Labels</Typography>}
        sx={{ mb: 0.5, ml: 0 }}
      />

      {/* Show Link Labels Toggle */}
      <FormControlLabel
        control={
          <Switch
            checked={showLinkLabels}
            onChange={(e) => setShowLinkLabels(e.target.checked)}
            size="small"
          />
        }
        label={<Typography variant="body2" sx={{ fontSize: '0.8rem' }}>Link Labels</Typography>}
        sx={{ mb: 1, ml: 0 }}
      />

      <Divider sx={{ my: 1.5 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          SIMULATION SETTINGS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Simulation
      </Typography>

      {/* Simulation Steps */}
      <Box sx={{ mt: 1.5, mb: 2 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Simulation Steps (N)
        </Typography>
        <TextField
          size="small"
          type="number"
          fullWidth
          value={simulationStepsInput}
          onChange={(e) => {
            // Update local input state - debounced effect will validate after 500ms
            setSimulationStepsInput(e.target.value)
          }}
          inputProps={{ step: 4 }}
          sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem', py: 0.75 } }}
          helperText={`Range: ${MIN_SIMULATION_STEPS}-${MAX_SIMULATION_STEPS} (auto-validates)`}
        />
      </Box>

      {/* Auto-Simulate Delay */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Auto-Simulate Delay (ms)
        </Typography>
        <TextField
          size="small"
          type="number"
          fullWidth
          value={autoSimulateDelayMs}
          onChange={(e) => {
            const val = parseInt(e.target.value) || DEFAULT_AUTO_SIMULATE_DELAY_MS
            setAutoSimulateDelayMs(Math.max(0, Math.min(1000, val)))
          }}
          inputProps={{ min: 0, max: 1000, step: 5 }}
          sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem', py: 0.75 } }}
          helperText="Delay before auto-simulation triggers"
        />
      </Box>

      {/* Trajectory Color Cycle */}
      <Box sx={{ mb: 2 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Trajectory Color Cycle
        </Typography>
        <FormControl fullWidth size="small">
          <Select
            value={trajectoryColorCycle}
            onChange={(e) => {
              setTrajectoryColorCycle(e.target.value as ColorCycleType)
              if (trajectoryData && autoSimulateEnabled) {
                triggerMechanismChange()
              }
            }}
            sx={{ fontSize: '0.85rem' }}
          >
            <MenuItem value="rainbow">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: '50%',
                  background: 'linear-gradient(90deg, red, orange, yellow, green, blue, violet)' }} />
                Rainbow
              </Box>
            </MenuItem>
            <MenuItem value="fire">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: '50%',
                  background: 'linear-gradient(90deg, #FA8112, #1A0A00, #FA8112)' }} />
                Fire
              </Box>
            </MenuItem>
            <MenuItem value="glow">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: '50%',
                  background: 'linear-gradient(90deg, #FA8112, #FFF8E8, #FA8112)' }} />
                Glow
              </Box>
            </MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Divider sx={{ my: 1.5 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          INTERACTION SETTINGS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Interaction
      </Typography>

      {/* Joint Merge Radius */}
      <Box sx={{ mt: 1.5, mb: 2 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Joint Merge Radius (units)
        </Typography>
        <TextField
          size="small"
          type="number"
          fullWidth
          value={jointMergeRadius}
          onChange={(e) => {
            const val = parseFloat(e.target.value) || DEFAULT_JOINT_MERGE_RADIUS
            setJointMergeRadius(Math.max(0.5, Math.min(20, val)))
          }}
          inputProps={{ min: 0.5, max: 20, step: 0.5 }}
          sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem', py: 0.75 } }}
          helperText="Snap distance for merging joints"
        />
      </Box>

      <Divider sx={{ my: 1.5 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          CANVAS/GRID SETTINGS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Canvas / Grid
      </Typography>

      {/* Canvas Background Color */}
      <Box sx={{ mt: 1.5, mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Canvas Background
        </Typography>
        <FormControl fullWidth size="small">
          <Select
            value={canvasBgColor}
            onChange={(e) => setCanvasBgColor(e.target.value as typeof canvasBgColor)}
            sx={{ fontSize: '0.85rem' }}
          >
            <MenuItem value="default">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: 1, bgcolor: darkMode ? '#1a1a1a' : '#fafafa', border: '1px solid #ccc' }} />
                Default
              </Box>
            </MenuItem>
            <MenuItem value="white">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: 1, bgcolor: '#ffffff', border: '1px solid #ccc' }} />
                White
              </Box>
            </MenuItem>
            <MenuItem value="cream">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: 1, bgcolor: '#FAF3E1', border: '1px solid #ccc' }} />
                Cream
              </Box>
            </MenuItem>
            <MenuItem value="dark">
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Box sx={{ width: 16, height: 16, borderRadius: 1, bgcolor: '#1a1a1a', border: '1px solid #ccc' }} />
                Dark
              </Box>
            </MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Grid Spacing - TODO */}
      <Box sx={{ mb: 1.5, opacity: 0.5 }}>
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.5 }}>
          Grid Spacing (units) <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
        </Typography>
        <FormControl fullWidth size="small" disabled>
          <Select value={20} sx={{ fontSize: '0.85rem' }}>
            <MenuItem value={5}>5 units</MenuItem>
            <MenuItem value={10}>10 units</MenuItem>
            <MenuItem value={20}>20 units</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Snap to Grid - TODO */}
      <FormControlLabel
        control={<Switch size="small" disabled />}
        label={
          <Typography variant="body2" sx={{ fontSize: '0.8rem', color: 'text.disabled' }}>
            Snap to Grid <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
          </Typography>
        }
        sx={{ mb: 1, ml: 0, opacity: 0.5 }}
      />

      <Divider sx={{ my: 1.5 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          VISUALIZATION SETTINGS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Visualization
      </Typography>

      {/* Joint Size */}
      <Box sx={{ mt: 1.5, mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Joint Size: {jointSize}px
        </Typography>
        <Box sx={{ px: 1 }}>
          <input
            type="range"
            min={3}
            max={16}
            value={jointSize}
            onChange={(e) => setJointSize(parseInt(e.target.value))}
            style={{ width: '100%', accentColor: '#FA8112' }}
          />
        </Box>
      </Box>

      {/* Link Thickness */}
      <Box sx={{ mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Link Thickness: {linkThickness}px
        </Typography>
        <Box sx={{ px: 1 }}>
          <input
            type="range"
            min={1}
            max={16}
            value={linkThickness}
            onChange={(e) => setLinkThickness(parseInt(e.target.value))}
            style={{ width: '100%', accentColor: '#FA8112' }}
          />
        </Box>
      </Box>

      {/* Trajectory Dot Size */}
      <Box sx={{ mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Trajectory Dot Size: {trajectoryDotSize}px
        </Typography>
        <Box sx={{ px: 1 }}>
          <input
            type="range"
            min={2}
            max={8}
            value={trajectoryDotSize}
            onChange={(e) => setTrajectoryDotSize(parseInt(e.target.value))}
            style={{ width: '100%', accentColor: '#FA8112' }}
          />
        </Box>
      </Box>

      {/* Selection Highlight Color */}
      <Box sx={{ mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Selection Highlight
        </Typography>
        <Box sx={{ display: 'flex', gap: 0.5 }}>
          {(['blue', 'orange', 'green', 'purple'] as const).map((color) => {
            const colorMap = { blue: '#1976d2', orange: '#FA8112', green: '#2e7d32', purple: '#9c27b0' }
            return (
              <Box
                key={color}
                onClick={() => setSelectionHighlightColor(color)}
                sx={{
                  width: 28, height: 28, borderRadius: 1,
                  bgcolor: colorMap[color],
                  cursor: 'pointer',
                  border: selectionHighlightColor === color ? '3px solid #fff' : '1px solid #ccc',
                  boxShadow: selectionHighlightColor === color ? `0 0 0 2px ${colorMap[color]}` : 'none',
                  '&:hover': { transform: 'scale(1.1)' },
                  transition: 'all 0.15s ease'
                }}
              />
            )
          })}
        </Box>
      </Box>

      {/* Show Measurement Units */}
      <FormControlLabel
        control={
          <Switch
            checked={showMeasurementUnits}
            onChange={(e) => setShowMeasurementUnits(e.target.checked)}
            size="small"
          />
        }
        label={<Typography variant="body2" sx={{ fontSize: '0.8rem' }}>Show Measurement Units</Typography>}
        sx={{ mb: 1, ml: 0 }}
      />

      <Divider sx={{ my: 1.5 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          ANIMATION SETTINGS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Animation
      </Typography>

      {/* Trajectory Style */}
      <Box sx={{ mt: 1.5, mb: 1.5 }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
          Trajectory Style
        </Typography>
        <FormControl fullWidth size="small">
          <Select
            value={trajectoryStyle}
            onChange={(e) => setTrajectoryStyle(e.target.value as typeof trajectoryStyle)}
            sx={{ fontSize: '0.85rem' }}
          >
            <MenuItem value="dots">Dots only</MenuItem>
            <MenuItem value="line">Line only</MenuItem>
            <MenuItem value="both">Dots + Line</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Animation Playback Speed - TODO */}
      <Box sx={{ mb: 1.5, opacity: 0.5 }}>
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.5 }}>
          Playback Speed <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
        </Typography>
        <FormControl fullWidth size="small" disabled>
          <Select value="1x" sx={{ fontSize: '0.85rem' }}>
            <MenuItem value="0.5x">0.5x</MenuItem>
            <MenuItem value="1x">1x</MenuItem>
            <MenuItem value="2x">2x</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Frame Interpolation - TODO */}
      <Box sx={{ mb: 1, opacity: 0.5 }}>
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.5 }}>
          Frame Interpolation <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
        </Typography>
        <FormControl fullWidth size="small" disabled>
          <Select value="linear" sx={{ fontSize: '0.85rem' }}>
            <MenuItem value="none">None</MenuItem>
            <MenuItem value="linear">Linear</MenuItem>
            <MenuItem value="smooth">Smooth</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Divider sx={{ my: 1.5 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          EXPORT/IMPORT SETTINGS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Export / Import
      </Typography>

      {/* Default File Format - TODO */}
      <Box sx={{ mt: 1.5, mb: 1.5, opacity: 0.5 }}>
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.5 }}>
          Default Format <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
        </Typography>
        <FormControl fullWidth size="small" disabled>
          <Select value="json" sx={{ fontSize: '0.85rem' }}>
            <MenuItem value="json">JSON</MenuItem>
            <MenuItem value="svg">SVG</MenuItem>
            <MenuItem value="png">PNG</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Auto-save Interval - TODO */}
      <Box sx={{ mb: 1.5, opacity: 0.5 }}>
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.5 }}>
          Auto-save Interval <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
        </Typography>
        <FormControl fullWidth size="small" disabled>
          <Select value="off" sx={{ fontSize: '0.85rem' }}>
            <MenuItem value="off">Off</MenuItem>
            <MenuItem value="30s">30 seconds</MenuItem>
            <MenuItem value="1m">1 minute</MenuItem>
            <MenuItem value="5m">5 minutes</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Include Metadata - TODO */}
      <FormControlLabel
        control={<Switch size="small" disabled defaultChecked />}
        label={
          <Typography variant="body2" sx={{ fontSize: '0.8rem', color: 'text.disabled' }}>
            Include Metadata <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
          </Typography>
        }
        sx={{ mb: 1, ml: 0, opacity: 0.5 }}
      />

      <Divider sx={{ my: 1.5 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          ADVANCED SETTINGS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Advanced
      </Typography>

      {/* Undo History Limit - TODO */}
      <Box sx={{ mt: 1.5, mb: 1.5, opacity: 0.5 }}>
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.5 }}>
          Undo History Limit <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
        </Typography>
        <FormControl fullWidth size="small" disabled>
          <Select value={50} sx={{ fontSize: '0.85rem' }}>
            <MenuItem value={10}>10 steps</MenuItem>
            <MenuItem value={50}>50 steps</MenuItem>
            <MenuItem value={100}>100 steps</MenuItem>
          </Select>
        </FormControl>
      </Box>

      {/* Debug Mode - TODO */}
      <FormControlLabel
        control={<Switch size="small" disabled />}
        label={
          <Typography variant="body2" sx={{ fontSize: '0.8rem', color: 'text.disabled' }}>
            Debug Mode <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
          </Typography>
        }
        sx={{ mb: 0.5, ml: 0, opacity: 0.5 }}
      />

      {/* Performance Mode - TODO */}
      <FormControlLabel
        control={<Switch size="small" disabled />}
        label={
          <Typography variant="body2" sx={{ fontSize: '0.8rem', color: 'text.disabled' }}>
            Performance Mode <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
          </Typography>
        }
        sx={{ mb: 0.5, ml: 0, opacity: 0.5 }}
      />

      {/* Collision Detection Tolerance - TODO */}
      <Box sx={{ mb: 1, opacity: 0.5 }}>
        <Typography variant="caption" sx={{ color: 'text.disabled', display: 'block', mb: 0.5 }}>
          Collision Tolerance <Chip label="TODO" size="small" sx={{ ml: 0.5, height: 16, fontSize: '0.6rem' }} />
        </Typography>
        <TextField
          size="small"
          type="number"
          fullWidth
          disabled
          defaultValue={0.1}
          sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem', py: 0.75 } }}
        />
      </Box>
    </Box>
  )

  // Render toolbar content based on id
  // ═══════════════════════════════════════════════════════════════════════════════
  // OPTIMIZATION TOOLBAR CONTENT
  // ═══════════════════════════════════════════════════════════════════════════════

  const OptimizationContent = () => {
    const hasCrank = canSimulate(pylinkDoc.pylinkage.joints)
    const selectedPath = targetPaths.find(p => p.id === selectedPathId)
    const canOptimize = selectedPath && selectedPath.targetJoint && hasCrank && !isOptimizing

    // Method descriptions for tooltips
    const methodDescriptions: Record<string, { name: string; description: string; pros: string; cons: string }> = {
      'pso': {
        name: 'Particle Swarm Optimization',
        description: 'Bio-inspired algorithm where particles explore the solution space, sharing information about good solutions.',
        pros: 'Robust, handles non-convex problems well, good at avoiding local minima',
        cons: 'Slower than gradient methods, requires tuning particles/iterations'
      },
      'pylinkage': {
        name: 'Pylinkage PSO',
        description: 'Native PSO implementation from the pylinkage library, optimized for linkage mechanisms.',
        pros: 'Designed specifically for linkages, well-tested',
        cons: 'Similar tradeoffs to standard PSO'
      },
      'scipy': {
        name: 'L-BFGS-B (SciPy)',
        description: 'Quasi-Newton method with bounded constraints. Uses gradient approximation for fast convergence.',
        pros: 'Very fast convergence for smooth problems',
        cons: 'Can get stuck in local minima, requires good initial guess'
      },
      'powell': {
        name: "Powell's Method",
        description: 'Direction-set method that minimizes along each coordinate direction sequentially.',
        pros: 'Gradient-free, good for noisy functions',
        cons: 'Slower than gradient methods, may not find global optimum'
      },
      'nelder-mead': {
        name: 'Nelder-Mead Simplex',
        description: 'Direct search method using a simplex of N+1 points that adapts to the function landscape.',
        pros: 'Very robust, no gradients needed, handles discontinuities',
        cons: 'Slow for high dimensions, local optimizer only'
      }
    }

    const currentMethodInfo = methodDescriptions[optMethod]

    return (
      <Box sx={{ p: 2 }}>
        {/* ═══════════════════════════════════════════════════════════════════════
            TARGET PATH SELECTION
            ═══════════════════════════════════════════════════════════════════════ */}
        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#e91e63', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
          <span>📍</span> Target Path
        </Typography>

        {targetPaths.length > 0 ? (
          <Box sx={{ mb: 2 }}>
            {targetPaths.map(path => (
              <Box
                key={path.id}
                onClick={() => setSelectedPathId(selectedPathId === path.id ? null : path.id)}
                sx={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  p: 1,
                  mb: 0.5,
                  borderRadius: 1,
                  cursor: 'pointer',
                  bgcolor: selectedPathId === path.id ? 'rgba(233, 30, 99, 0.15)' : 'rgba(0,0,0,0.02)',
                  border: '2px solid',
                  borderColor: selectedPathId === path.id ? '#e91e63' : 'transparent',
                  transition: 'all 0.15s ease',
                  '&:hover': { bgcolor: 'rgba(233, 30, 99, 0.08)' }
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Box sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: path.color }} />
                  <Box>
                    <Typography variant="body2" sx={{ fontWeight: selectedPathId === path.id ? 600 : 400 }}>
                      {path.name}
                    </Typography>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                      {path.points.length} points
                      {path.targetJoint && ` • ${path.targetJoint}`}
                    </Typography>
                  </Box>
                </Box>
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation()
                    setTargetPaths(prev => prev.filter(p => p.id !== path.id))
                    if (selectedPathId === path.id) setSelectedPathId(null)
                  }}
                  sx={{ width: 24, height: 24, color: '#999', '&:hover': { color: '#d32f2f' } }}
                >
                  ×
                </IconButton>
              </Box>
            ))}
          </Box>
        ) : (
          <Box sx={{
            p: 2,
            mb: 2,
            borderRadius: 1,
            bgcolor: 'rgba(0,0,0,0.03)',
            border: '1px dashed rgba(0,0,0,0.2)',
            textAlign: 'center'
          }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
              No target paths yet
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary', fontStyle: 'italic' }}>
              Use <strong>Draw Path</strong> tool (T) to create one
            </Typography>
          </Box>
        )}

        {/* Joint selector */}
        {selectedPathId && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
              Joint to Optimize
            </Typography>
            <FormControl size="small" fullWidth>
              <Select
                value={selectedPath?.targetJoint || ''}
                onChange={(e) => {
                  setTargetPaths(prev => prev.map(p =>
                    p.id === selectedPathId ? { ...p, targetJoint: e.target.value as string } : p
                  ))
                }}
                displayEmpty
                sx={{ fontSize: '0.85rem' }}
              >
                <MenuItem value="" sx={{ fontSize: '0.85rem' }}>
                  <em>Select joint...</em>
                </MenuItem>
                {pylinkDoc.pylinkage.joints
                  .filter(j => j.type === 'Crank' || j.type === 'Revolute')
                  .map(j => (
                    <MenuItem key={j.name} value={j.name} sx={{ fontSize: '0.85rem' }}>
                      {j.name} <Chip label={j.type} size="small" sx={{ ml: 1, height: 18, fontSize: '0.65rem' }} />
                    </MenuItem>
                  ))
                }
              </Select>
            </FormControl>
          </Box>
        )}

        {/* ═══════════════════════════════════════════════════════════════════════
            TRAJECTORY PREPROCESSING
            ═══════════════════════════════════════════════════════════════════════ */}
        {selectedPathId && selectedPath && (
          <>
            <Divider sx={{ my: 2 }} />

            <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#00897b', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
              <span>🔄</span> Path Preprocessing
            </Typography>

            <Box sx={{
              p: 1.5,
              mb: 1.5,
              borderRadius: 1,
              bgcolor: 'rgba(0, 137, 123, 0.05)',
              border: '1px solid rgba(0, 137, 123, 0.2)'
            }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1 }}>
                Current path: <strong>{selectedPath.points.length} points</strong>
                {preprocessResult && (
                  <> • Processed from {preprocessResult.originalPoints} points</>
                )}
              </Typography>

              {/* Smoothing Section */}
              <Box sx={{ mb: 1.5 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={prepEnableSmooth}
                      onChange={(e) => setPrepEnableSmooth(e.target.checked)}
                      size="small"
                      color="primary"
                    />
                  }
                  label={<Typography variant="caption" sx={{ fontWeight: 500 }}>Enable Smoothing</Typography>}
                />

                {prepEnableSmooth && (
                  <Box sx={{ pl: 1, mt: 0.5 }}>
                    <Box sx={{ display: 'flex', gap: 1, mb: 1 }}>
                      <Box sx={{ flex: 1 }}>
                        <Tooltip title="Smoothing filter type. Savgol preserves peaks, Moving Avg is aggressive, Gaussian is natural." placement="top">
                          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.25, cursor: 'help', fontSize: '0.65rem' }}>
                            Method ⓘ
                          </Typography>
                        </Tooltip>
                        <Select
                          size="small"
                          fullWidth
                          value={prepSmoothMethod}
                          onChange={(e) => setPrepSmoothMethod(e.target.value as typeof prepSmoothMethod)}
                          sx={{ fontSize: '0.75rem', '& .MuiSelect-select': { py: 0.5 } }}
                        >
                          <MenuItem value="savgol" sx={{ fontSize: '0.75rem' }}>Savitzky-Golay</MenuItem>
                          <MenuItem value="moving_avg" sx={{ fontSize: '0.75rem' }}>Moving Average</MenuItem>
                          <MenuItem value="gaussian" sx={{ fontSize: '0.75rem' }}>Gaussian</MenuItem>
                        </Select>
                      </Box>
                    </Box>

                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Box sx={{ flex: 1 }}>
                        <Tooltip title="Window size. Larger = more smoothing. 2-4: light, 8-16: medium, 32+: heavy" placement="top">
                          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.25, cursor: 'help', fontSize: '0.65rem' }}>
                            Window ⓘ
                          </Typography>
                        </Tooltip>
                        <Select
                          size="small"
                          fullWidth
                          value={prepSmoothWindow}
                          onChange={(e) => setPrepSmoothWindow(e.target.value as number)}
                          sx={{ fontSize: '0.75rem', '& .MuiSelect-select': { py: 0.5 } }}
                        >
                          <MenuItem value={2} sx={{ fontSize: '0.75rem' }}>2 (Light)</MenuItem>
                          <MenuItem value={4} sx={{ fontSize: '0.75rem' }}>4 (Default)</MenuItem>
                          <MenuItem value={8} sx={{ fontSize: '0.75rem' }}>8 (Medium)</MenuItem>
                          <MenuItem value={16} sx={{ fontSize: '0.75rem' }}>16</MenuItem>
                          <MenuItem value={32} sx={{ fontSize: '0.75rem' }}>32 (Heavy)</MenuItem>
                          <MenuItem value={64} sx={{ fontSize: '0.75rem' }}>64 (Max)</MenuItem>
                        </Select>
                      </Box>

                      <Box sx={{ flex: 1 }}>
                        <Tooltip title="Polynomial order for Savgol. Must be < window. Higher = preserves peaks better." placement="top">
                          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.25, cursor: 'help', fontSize: '0.65rem' }}>
                            Polyorder ⓘ
                          </Typography>
                        </Tooltip>
                        <Select
                          size="small"
                          fullWidth
                          value={prepSmoothPolyorder}
                          onChange={(e) => setPrepSmoothPolyorder(e.target.value as number)}
                          disabled={prepSmoothMethod !== 'savgol'}
                          sx={{ fontSize: '0.75rem', '& .MuiSelect-select': { py: 0.5 } }}
                        >
                          <MenuItem value={1} sx={{ fontSize: '0.75rem' }}>1 (Linear)</MenuItem>
                          <MenuItem value={2} sx={{ fontSize: '0.75rem' }}>2</MenuItem>
                          <MenuItem value={3} sx={{ fontSize: '0.75rem' }}>3 (Default)</MenuItem>
                          <MenuItem value={4} sx={{ fontSize: '0.75rem' }}>4</MenuItem>
                          <MenuItem value={5} sx={{ fontSize: '0.75rem' }}>5</MenuItem>
                        </Select>
                      </Box>
                    </Box>
                  </Box>
                )}
              </Box>

              {/* Resampling Section */}
              <Box sx={{ mb: 1.5 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={prepEnableResample}
                      onChange={(e) => setPrepEnableResample(e.target.checked)}
                      size="small"
                      color="primary"
                    />
                  }
                  label={<Typography variant="caption" sx={{ fontWeight: 500 }}>Enable Resampling</Typography>}
                />

                {prepEnableResample && (
                  <Box sx={{ pl: 1, mt: 0.5 }}>
                    <Box sx={{ display: 'flex', gap: 1 }}>
                      <Box sx={{ flex: 1 }}>
                        <Tooltip title={`Target number of points. Uses current Simulation Steps (${simulationSteps}) for optimization consistency.`} placement="top">
                          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.25, cursor: 'help', fontSize: '0.65rem' }}>
                            Target Points ⓘ
                          </Typography>
                        </Tooltip>
                        <TextField
                          type="number"
                          size="small"
                          fullWidth
                          value={prepTargetNSteps}
                          onChange={(e) => {
                            const val = parseInt(e.target.value)
                            if (!isNaN(val) && val >= 4 && val <= 256) {
                              setPrepTargetNSteps(val)
                            }
                          }}
                          inputProps={{ min: 4, max: 256, step: 4 }}
                          helperText={simulationSteps !== prepTargetNSteps ? `Sim uses ${simulationSteps}` : undefined}
                          sx={{
                            '& .MuiInputBase-input': { fontSize: '0.75rem', py: 0.5 },
                            '& .MuiFormHelperText-root': { fontSize: '0.6rem', mt: 0.25, color: 'warning.main' }
                          }}
                        />
                      </Box>

                      <Box sx={{ flex: 1 }}>
                        <Tooltip title="Interpolation method. Parametric is best for closed curves." placement="top">
                          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.25, cursor: 'help', fontSize: '0.65rem' }}>
                            Method ⓘ
                          </Typography>
                        </Tooltip>
                        <Select
                          size="small"
                          fullWidth
                          value={prepResampleMethod}
                          onChange={(e) => setPrepResampleMethod(e.target.value as typeof prepResampleMethod)}
                          sx={{ fontSize: '0.75rem', '& .MuiSelect-select': { py: 0.5 } }}
                        >
                          <MenuItem value="parametric" sx={{ fontSize: '0.75rem' }}>Parametric</MenuItem>
                          <MenuItem value="cubic" sx={{ fontSize: '0.75rem' }}>Cubic</MenuItem>
                          <MenuItem value="linear" sx={{ fontSize: '0.75rem' }}>Linear</MenuItem>
                        </Select>
                      </Box>
                    </Box>
                  </Box>
                )}
              </Box>

              {/* Preprocess Button */}
              <Button
                variant="outlined"
                fullWidth
                size="small"
                onClick={preprocessTrajectory}
                disabled={isPreprocessing || (!prepEnableSmooth && !prepEnableResample)}
                sx={{
                  textTransform: 'none',
                  fontSize: '0.8rem',
                  color: '#00897b',
                  borderColor: '#00897b',
                  '&:hover': {
                    borderColor: '#00695c',
                    bgcolor: 'rgba(0, 137, 123, 0.08)'
                  },
                  '&.Mui-disabled': { borderColor: '#ccc' }
                }}
              >
                {isPreprocessing ? '⏳ Processing...' : '🔄 Apply Preprocessing'}
              </Button>

              {/* Preprocessing Result */}
              {preprocessResult && (
                <Box sx={{ mt: 1, p: 1, borderRadius: 0.5, bgcolor: 'rgba(0, 137, 123, 0.1)' }}>
                  <Typography variant="caption" sx={{ display: 'block', color: '#00695c', fontWeight: 500 }}>
                    ✓ Processed successfully
                  </Typography>
                  <Typography variant="caption" sx={{ display: 'block', color: 'text.secondary', fontSize: '0.65rem' }}>
                    {preprocessResult.originalPoints} → {preprocessResult.outputPoints} points
                    {preprocessResult.analysis && (
                      <>
                        {' • '}Path length: {(preprocessResult.analysis.total_path_length as number)?.toFixed(1)}
                        {preprocessResult.analysis.is_closed && ' • Closed curve'}
                      </>
                    )}
                  </Typography>
                </Box>
              )}
            </Box>
          </>
        )}

        <Divider sx={{ my: 2 }} />

        {/* ═══════════════════════════════════════════════════════════════════════
            SIMULATION STEPS (N_STEPS)
            ═══════════════════════════════════════════════════════════════════════ */}
        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#6a1b9a', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
          <span>📊</span> Simulation Steps
        </Typography>

        <Box sx={{ mb: 2 }}>
          <Tooltip title="Number of trajectory points for simulation and optimization. Higher = more precision but slower. Should match preprocessed trajectory points." placement="right">
            <TextField
              type="number"
              size="small"
              fullWidth
              label="N_STEPS"
              value={simulationStepsInput}
              onChange={(e) => setSimulationStepsInput(e.target.value)}
              inputProps={{ min: MIN_SIMULATION_STEPS, max: MAX_SIMULATION_STEPS, step: 4 }}
              helperText={`Range: ${MIN_SIMULATION_STEPS}-${MAX_SIMULATION_STEPS}. Current: ${simulationSteps}`}
              sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
            />
          </Tooltip>

          {/* Sync button to match Target Points with Simulation Steps */}
          {prepTargetNSteps !== simulationSteps && (
            <Button
              size="small"
              variant="text"
              onClick={() => setPrepTargetNSteps(simulationSteps)}
              sx={{
                mt: 0.5,
                textTransform: 'none',
                fontSize: '0.7rem',
                color: '#6a1b9a'
              }}
            >
              ↻ Sync preprocessing target ({prepTargetNSteps}) to {simulationSteps}
            </Button>
          )}
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* ═══════════════════════════════════════════════════════════════════════
            OPTIMIZATION METHOD
            ═══════════════════════════════════════════════════════════════════════ */}
        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#1976d2', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
          <span>🔧</span> Method
        </Typography>

        <FormControl size="small" fullWidth sx={{ mb: 1 }}>
          <Select
            value={optMethod}
            onChange={(e) => setOptMethod(e.target.value as typeof optMethod)}
            sx={{ fontSize: '0.85rem' }}
          >
            <MenuItem value="pso">Particle Swarm (PSO)</MenuItem>
            <MenuItem value="pylinkage">Pylinkage PSO</MenuItem>
            <MenuItem value="scipy">L-BFGS-B (SciPy)</MenuItem>
            <MenuItem value="powell">Powell's Method</MenuItem>
            <MenuItem value="nelder-mead">Nelder-Mead Simplex</MenuItem>
          </Select>
        </FormControl>

        {/* Method description */}
        <Box sx={{
          p: 1.5,
          mb: 2,
          borderRadius: 1,
          bgcolor: 'rgba(25, 118, 210, 0.05)',
          border: '1px solid rgba(25, 118, 210, 0.2)'
        }}>
          <Typography variant="caption" sx={{ fontWeight: 600, color: '#1976d2', display: 'block' }}>
            {currentMethodInfo.name}
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.5 }}>
            {currentMethodInfo.description}
          </Typography>
          <Box sx={{ mt: 1, display: 'flex', gap: 2 }}>
            <Box sx={{ flex: 1 }}>
              <Typography variant="caption" sx={{ color: '#2e7d32', fontWeight: 500 }}>✓ Pros</Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', fontSize: '0.65rem' }}>
                {currentMethodInfo.pros}
              </Typography>
            </Box>
            <Box sx={{ flex: 1 }}>
              <Typography variant="caption" sx={{ color: '#d32f2f', fontWeight: 500 }}>✗ Cons</Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', fontSize: '0.65rem' }}>
                {currentMethodInfo.cons}
              </Typography>
            </Box>
          </Box>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* ═══════════════════════════════════════════════════════════════════════
            HYPERPARAMETERS
            ═══════════════════════════════════════════════════════════════════════ */}
        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#7b1fa2', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
          <span>⚙️</span> Hyperparameters
        </Typography>

        {/* PSO-specific parameters */}
        {(optMethod === 'pso' || optMethod === 'pylinkage') && (
          <>
            <Box sx={{ mb: 1.5 }}>
              <Tooltip title="Number of particles in the swarm. More particles = better exploration but slower. Typical: 20-50" placement="left">
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
                  Swarm Size (Particles) ⓘ
                </Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                fullWidth
                value={optNParticles}
                onChange={(e) => setOptNParticles(Math.max(5, Math.min(1024, parseInt(e.target.value) || 32)))}
                inputProps={{ min: 5, max: 1024, step: 16 }}
                sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Box>

            <Box sx={{ mb: 1.5 }}>
              <Tooltip title="Number of iterations for the swarm. More iterations = better convergence but slower. Typical: 256-1024, max 10000" placement="left">
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
                  Iterations ⓘ
                </Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                fullWidth
                value={optIterations}
                onChange={(e) => setOptIterations(Math.max(10, Math.min(10000, parseInt(e.target.value) || 512)))}
                inputProps={{ min: 10, max: 10000, step: 64 }}
                sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Box>
          </>
        )}

        {/* SciPy-specific parameters */}
        {(optMethod === 'scipy' || optMethod === 'powell' || optMethod === 'nelder-mead') && (
          <>
            <Box sx={{ mb: 1.5 }}>
              <Tooltip title="Maximum number of function evaluations. Prevents infinite loops. Typical: 100-1000" placement="left">
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
                  Max Iterations ⓘ
                </Typography>
              </Tooltip>
              <TextField
                type="number"
                size="small"
                fullWidth
                value={optMaxIterations}
                onChange={(e) => setOptMaxIterations(Math.max(10, Math.min(10000, parseInt(e.target.value) || 100)))}
                inputProps={{ min: 10, max: 10000, step: 50 }}
                sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
              />
            </Box>

            <Box sx={{ mb: 1.5 }}>
              <Tooltip title="Convergence tolerance. Smaller = more precise but slower. Typical: 1e-4 to 1e-8" placement="left">
                <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
                  Tolerance ⓘ
                </Typography>
              </Tooltip>
              <Select
                size="small"
                fullWidth
                value={optTolerance}
                onChange={(e) => setOptTolerance(e.target.value as number)}
                sx={{ fontSize: '0.85rem' }}
              >
                <MenuItem value={1e-4}>1e-4 (Fast, less precise)</MenuItem>
                <MenuItem value={1e-5}>1e-5</MenuItem>
                <MenuItem value={1e-6}>1e-6 (Default)</MenuItem>
                <MenuItem value={1e-7}>1e-7</MenuItem>
                <MenuItem value={1e-8}>1e-8 (Slow, very precise)</MenuItem>
              </Select>
            </Box>
          </>
        )}

        <Divider sx={{ my: 2 }} />

        {/* ═══════════════════════════════════════════════════════════════════════
            BOUNDS & CONSTRAINTS
            ═══════════════════════════════════════════════════════════════════════ */}
        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: '#ed6c02', mb: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
          <span>📐</span> Bounds
        </Typography>

        <Box sx={{ mb: 1.5 }}>
          <Tooltip title="How much link lengths can vary from initial values. Factor of 2.0 means lengths can be 0.5x to 2.0x original" placement="left">
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
              Bounds Factor ⓘ
            </Typography>
          </Tooltip>
          <Select
            size="small"
            fullWidth
            value={optBoundsFactor}
            onChange={(e) => setOptBoundsFactor(e.target.value as number)}
            sx={{ fontSize: '0.85rem' }}
          >
            <MenuItem value={1.25}>±25% (Conservative)</MenuItem>
            <MenuItem value={1.5}>±50%</MenuItem>
            <MenuItem value={2.0}>±100% (Default)</MenuItem>
            <MenuItem value={3.0}>±200% (Wide)</MenuItem>
            <MenuItem value={5.0}>±400% (Very Wide)</MenuItem>
          </Select>
        </Box>

        <Box sx={{ mb: 1.5 }}>
          <Tooltip title="Minimum allowed link length to prevent degenerate solutions" placement="left">
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5, cursor: 'help' }}>
              Min Link Length ⓘ
            </Typography>
          </Tooltip>
          <TextField
            type="number"
            size="small"
            fullWidth
            value={optMinLength}
            onChange={(e) => setOptMinLength(Math.max(0.01, parseFloat(e.target.value) || 0.1))}
            inputProps={{ min: 0.01, max: 10, step: 0.1 }}
            sx={{ '& .MuiInputBase-input': { fontSize: '0.85rem' } }}
          />
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* ═══════════════════════════════════════════════════════════════════════
            RUN OPTIMIZATION
            ═══════════════════════════════════════════════════════════════════════ */}
        <Box sx={{ mb: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={optVerbose}
                onChange={(e) => setOptVerbose(e.target.checked)}
                size="small"
              />
            }
            label={<Typography variant="caption">Verbose logging</Typography>}
          />
        </Box>

        <Button
          variant="contained"
          fullWidth
          size="large"
          onClick={runOptimization}
          disabled={!canOptimize}
          sx={{
            textTransform: 'none',
            fontSize: '1rem',
            fontWeight: 600,
            py: 1.5,
            backgroundColor: '#e91e63',
            '&:hover': { backgroundColor: '#c2185b' },
            '&.Mui-disabled': { backgroundColor: '#e0e0e0' }
          }}
        >
          {isOptimizing ? (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <span>⏳</span> Optimizing...
            </Box>
          ) : (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <span>⚡</span> Run Optimization
            </Box>
          )}
        </Button>

        {/* Disabled reason */}
        {!canOptimize && !isOptimizing && (
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 1, textAlign: 'center' }}>
            {!hasCrank ? 'Need a valid mechanism with Crank' :
             !selectedPath ? 'Select a target path above' :
             !selectedPath.targetJoint ? 'Select a joint to optimize' :
             'Ready to optimize'}
          </Typography>
        )}

        {/* Warning if path points don't match simulation steps */}
        {canOptimize && selectedPath && selectedPath.points.length !== simulationSteps && (
          <Box sx={{
            mt: 1,
            p: 1,
            borderRadius: 1,
            bgcolor: 'rgba(237, 108, 2, 0.1)',
            border: '1px solid #ed6c02'
          }}>
            <Typography variant="caption" sx={{ color: '#ed6c02', display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <span>⚠️</span> Path has {selectedPath.points.length} points but simulation uses {simulationSteps}.
              Preprocess or adjust N_STEPS for best results.
            </Typography>
          </Box>
        )}

        {/* ═══════════════════════════════════════════════════════════════════════
            RESULTS
            ═══════════════════════════════════════════════════════════════════════ */}
        {optimizationResult && (
          <Box sx={{ mt: 2 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, color: optimizationResult.success ? '#2e7d32' : '#d32f2f' }}>
                {optimizationResult.success ? '✓ Results' : '✗ Failed'}
              </Typography>
              {preOptimizationDoc && optimizationResult.success && (
                <Button
                  size="small"
                  variant="outlined"
                  color="warning"
                  onClick={revertOptimization}
                  sx={{
                    textTransform: 'none',
                    fontSize: '0.7rem',
                    py: 0.25,
                    px: 1,
                    minWidth: 'auto'
                  }}
                >
                  ↩ Revert
                </Button>
              )}
            </Box>

            <Box sx={{
              p: 1.5,
              borderRadius: 1,
              bgcolor: optimizationResult.success ? 'rgba(46, 125, 50, 0.08)' : 'rgba(211, 47, 47, 0.08)',
              border: '1px solid',
              borderColor: optimizationResult.success ? '#4caf50' : '#f44336'
            }}>
              {optimizationResult.success ? (
                <>
                  {/* Error metrics */}
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>Initial Error</Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 600, color: '#ff7043' }}>
                      {optimizationResult.initialError.toFixed(4)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>Final Error</Typography>
                    <Typography variant="caption" sx={{ fontFamily: 'monospace', fontWeight: 600, color: '#2e7d32' }}>
                      {optimizationResult.finalError.toFixed(4)}
                    </Typography>
                  </Box>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="caption" sx={{ color: 'text.secondary' }}>Improvement</Typography>
                    <Typography variant="caption" sx={{
                      fontWeight: 700,
                      color: optimizationResult.initialError > 0
                        ? ((1 - optimizationResult.finalError / optimizationResult.initialError) * 100 > 50 ? '#2e7d32' : '#ed6c02')
                        : '#666'
                    }}>
                      {optimizationResult.initialError > 0
                        ? `${((1 - optimizationResult.finalError / optimizationResult.initialError) * 100).toFixed(1)}%`
                        : 'N/A'}
                    </Typography>
                  </Box>
                  {optimizationResult.iterations && (
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>Iterations</Typography>
                      <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                        {optimizationResult.iterations}
                      </Typography>
                    </Box>
                  )}
                  {optimizationResult.executionTimeMs && (
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>Time</Typography>
                      <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                        {(optimizationResult.executionTimeMs / 1000).toFixed(2)}s
                      </Typography>
                    </Box>
                  )}

                  {/* Show dimension changes: Before → After */}
                  {optimizationResult.optimizedDimensions && optimizationResult.originalDimensions &&
                   Object.keys(optimizationResult.optimizedDimensions).length > 0 && (
                    <Box sx={{ mt: 1.5, pt: 1.5, borderTop: '1px solid rgba(0,0,0,0.1)' }}>
                      <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 1 }}>
                        Dimension Changes
                      </Typography>
                      <Box sx={{
                        display: 'grid',
                        gridTemplateColumns: '1fr auto auto auto',
                        gap: 0.5,
                        fontSize: '0.65rem',
                        '& > *': { py: 0.25 }
                      }}>
                        {/* Header */}
                        <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', fontSize: '0.6rem' }}>
                          Parameter
                        </Typography>
                        <Typography variant="caption" sx={{ fontWeight: 600, color: '#ff7043', fontSize: '0.6rem', textAlign: 'right' }}>
                          Before
                        </Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary', fontSize: '0.6rem', textAlign: 'center' }}>
                          →
                        </Typography>
                        <Typography variant="caption" sx={{ fontWeight: 600, color: '#2e7d32', fontSize: '0.6rem', textAlign: 'right' }}>
                          After
                        </Typography>

                        {/* Data rows */}
                        {Object.entries(optimizationResult.optimizedDimensions).map(([name, newValue]) => {
                          const oldValue = optimizationResult.originalDimensions?.[name] ?? newValue
                          const changed = Math.abs((newValue as number) - (oldValue as number)) > 0.001
                          return (
                            <React.Fragment key={name}>
                              <Typography variant="caption" sx={{
                                color: 'text.secondary',
                                fontSize: '0.65rem',
                                fontWeight: changed ? 500 : 400
                              }}>
                                {name.replace(/_/g, ' ')}
                              </Typography>
                              <Typography variant="caption" sx={{
                                fontFamily: 'monospace',
                                fontSize: '0.65rem',
                                textAlign: 'right',
                                color: '#ff7043'
                              }}>
                                {(oldValue as number).toFixed(2)}
                              </Typography>
                              <Typography variant="caption" sx={{
                                color: changed ? '#1976d2' : 'text.secondary',
                                fontSize: '0.65rem',
                                textAlign: 'center'
                              }}>
                                {changed ? '→' : '='}
                              </Typography>
                              <Typography variant="caption" sx={{
                                fontFamily: 'monospace',
                                fontSize: '0.65rem',
                                textAlign: 'right',
                                color: '#2e7d32',
                                fontWeight: changed ? 600 : 400
                              }}>
                                {(newValue as number).toFixed(2)}
                              </Typography>
                            </React.Fragment>
                          )
                        })}
                      </Box>
                    </Box>
                  )}

                  {/* Canvas updated notice */}
                  <Box sx={{
                    mt: 1.5,
                    p: 1,
                    bgcolor: 'rgba(25, 118, 210, 0.1)',
                    borderRadius: 1,
                    border: '1px solid rgba(25, 118, 210, 0.3)'
                  }}>
                    <Typography variant="caption" sx={{ color: '#1976d2', fontWeight: 500 }}>
                      ✓ Canvas updated with optimized dimensions
                    </Typography>
                    {preOptimizationDoc && (
                      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.5, fontSize: '0.65rem' }}>
                        Click "Revert" to restore original dimensions
                      </Typography>
                    )}
                  </Box>
                </>
              ) : (
                <Typography variant="caption" sx={{ color: '#d32f2f' }}>
                  {optimizationResult.message}
                </Typography>
              )}
            </Box>
          </Box>
        )}
      </Box>
    )
  }

  const renderToolbarContent = (id: string) => {
    switch (id) {
      case 'tools': return <ToolsContent />
      case 'links': return <LinksContent />
      case 'nodes': return <NodesContent />
      case 'more': return <MoreContent />
      case 'optimize': return <OptimizationContent />
      case 'settings': return <SettingsContent />
      default: return null
    }
  }

  // Demo: load a 4-bar linkage
  const loadDemo4Bar = () => {
    // 4-bar demo: twice the original size, positioned near (90, 90)
    // Pre-calculate positions for non-Static joints (single source of truth)
    const crankAnchorPos: [number, number] = [90, 90]
    const rockerAnchorPos: [number, number] = [150, 90]
    const crankAngle = 0.2618
    const crankDistance = 20
    const crankPos: [number, number] = [
      crankAnchorPos[0] + crankDistance * Math.cos(crankAngle),
      crankAnchorPos[1] + crankDistance * Math.sin(crankAngle)
    ]

    // Calculate Revolute position using circle-circle intersection
    const d0 = 50  // distance from crank
    const d1 = 40  // distance from rocker_anchor
    const dx = rockerAnchorPos[0] - crankPos[0]
    const dy = rockerAnchorPos[1] - crankPos[1]
    const d = Math.sqrt(dx * dx + dy * dy)
    const a = (d0 * d0 - d1 * d1 + d * d) / (2 * d)
    const h = Math.sqrt(Math.max(0, d0 * d0 - a * a))
    const px = crankPos[0] + (a * dx) / d
    const py = crankPos[1] + (a * dy) / d
    const couplerJointPos: [number, number] = [
      px - (h * dy) / d,  // One of two solutions (above the line)
      py + (h * dx) / d
    ]

    const demo: PylinkDocument = {
      name: '4bar',
      pylinkage: {
        name: '4bar',
        joints: [
          { type: 'Static', name: 'crank_anchor', x: crankAnchorPos[0], y: crankAnchorPos[1] },
          { type: 'Static', name: 'rocker_anchor', x: rockerAnchorPos[0], y: rockerAnchorPos[1] },
          { type: 'Crank', name: 'crank', joint0: { ref: 'crank_anchor' }, distance: crankDistance, angle: crankAngle },
          { type: 'Revolute', name: 'coupler_rocker_joint', joint0: { ref: 'crank' }, joint1: { ref: 'rocker_anchor' }, distance0: d0, distance1: d1 }
        ],
        solve_order: ['crank_anchor', 'rocker_anchor', 'crank', 'coupler_rocker_joint']
      },
      meta: {
        joints: {
          // Store UI positions for non-Static joints (single source of truth)
          // show_path enables trajectory display for moving joints
          crank: { color: '#ff7f0e', zlevel: 0, x: crankPos[0], y: crankPos[1], show_path: true },
          coupler_rocker_joint: { color: '#2ca02c', zlevel: 0, x: couplerJointPos[0], y: couplerJointPos[1], show_path: true }
        },
        links: {
          ground: { color: '#7f7f7f', connects: ['crank_anchor', 'rocker_anchor'], isGround: true },
          crank_link: { color: '#ff7f0e', connects: ['crank_anchor', 'crank'] },
          coupler: { color: '#2ca02c', connects: ['crank', 'coupler_rocker_joint'] },
          rocker: { color: '#1f77b4', connects: ['coupler_rocker_joint', 'rocker_anchor'] }
        }
      }
    }
    setPylinkDoc(demo)
    clearTrajectory()  // Clear any old trajectory
    showStatus('Loaded demo 4-bar linkage', 'success', 2000)
  }

  // Save pylink graph to server
  const savePylinkGraph = async () => {
    try {
      showStatus('Saving...', 'action')
      const response = await fetch('/api/save-pylink-graph', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pylinkDoc)
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success') {
        showStatus(`Saved as ${result.filename}`, 'success', 3000)
      } else {
        showStatus(result.message || 'Save failed', 'error', 3000)
      }
    } catch (error) {
      showStatus(`Save error: ${error}`, 'error', 3000)
    }
  }

  // Validate mechanism - calls backend to check if mechanism can be simulated
  const validateMechanism = async () => {
    try {
      showStatus('Validating mechanism...', 'action')
      const response = await fetch('/api/validate-mechanism', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pylinkDoc)
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'error') {
        showStatus(`Validation error: ${result.message}`, 'error', 5000)
        return
      }

      // Handle validation result
      if (result.valid) {
        const groupCount = result.valid_groups?.length || 0
        const jointCount = result.valid_groups?.reduce((acc: number, g: { joints: string[] }) => acc + g.joints.length, 0) || 0
        showStatus(`✓ Valid mechanism: ${groupCount} group(s), ${jointCount} joints`, 'success', 4000)
      } else {
        // Collect error messages
        const errors = result.errors || []
        const errorMsg = errors.length > 0
          ? errors.slice(0, 2).join('; ') + (errors.length > 2 ? ` (+${errors.length - 2} more)` : '')
          : 'No valid mechanism groups found'
        showStatus(`⚠ Invalid: ${errorMsg}`, 'warning', 5000)
      }
    } catch (error) {
      showStatus(`Validation error: ${error}`, 'error', 3000)
    }
  }

  // Repair Revolute joints: sync joint0/joint1 references with visual links
  // This fixes the bug where joints have incorrect parent references
  const repairRevoluteJointReferences = useCallback((doc: PylinkDocument): PylinkDocument => {
    let repairCount = 0
    const repairedJoints = doc.pylinkage.joints.map(joint => {
      if (joint.type !== 'Revolute') return joint

      // Find joints connected via visual links
      const connectedJoints = getConnectedJointsFromLinks(joint.name, doc.meta.links)

      if (connectedJoints.length >= 2) {
        const currentParent0 = joint.joint0.ref
        const currentParent1 = joint.joint1.ref

        // Check if current parents match the visual links
        const needsRepair = !connectedJoints.includes(currentParent0) ||
                           !connectedJoints.includes(currentParent1)

        if (needsRepair) {
          // Get position for distance calculations
          const pos = doc.meta.joints[joint.name]?.x !== undefined && doc.meta.joints[joint.name]?.y !== undefined
            ? [doc.meta.joints[joint.name].x!, doc.meta.joints[joint.name].y!] as [number, number]
            : null

          if (pos) {
            const newParent0 = connectedJoints[0]
            const newParent1 = connectedJoints[1]

            // Get parent positions
            const parent0Joint = doc.pylinkage.joints.find(j => j.name === newParent0)
            const parent1Joint = doc.pylinkage.joints.find(j => j.name === newParent1)

            const parent0Pos = parent0Joint?.type === 'Static'
              ? [parent0Joint.x, parent0Joint.y] as [number, number]
              : doc.meta.joints[newParent0]?.x !== undefined
                ? [doc.meta.joints[newParent0].x!, doc.meta.joints[newParent0].y!] as [number, number]
                : null

            const parent1Pos = parent1Joint?.type === 'Static'
              ? [parent1Joint.x, parent1Joint.y] as [number, number]
              : doc.meta.joints[newParent1]?.x !== undefined
                ? [doc.meta.joints[newParent1].x!, doc.meta.joints[newParent1].y!] as [number, number]
                : null

            if (parent0Pos && parent1Pos) {
              repairCount++
              console.log(`Repaired ${joint.name}: ${currentParent0},${currentParent1} → ${newParent0},${newParent1}`)
              return {
                ...joint,
                joint0: { ref: newParent0 },
                joint1: { ref: newParent1 },
                distance0: calculateDistance(parent0Pos, pos),
                distance1: calculateDistance(parent1Pos, pos)
              } as RevoluteJoint
            }
          }
        }
      }
      return joint
    })

    if (repairCount > 0) {
      showStatus(`Repaired ${repairCount} Revolute joint reference(s)`, 'info', 3000)
    }

    return {
      ...doc,
      pylinkage: {
        ...doc.pylinkage,
        joints: repairedJoints
      }
    }
  }, [getConnectedJointsFromLinks, showStatus])

  // Synchronize meta.joints positions with pylinkage.joints distances by running simulation
  // This ensures the visual representation matches the actual mechanism geometry
  const syncPositionsWithDistances = async (doc: PylinkDocument): Promise<PylinkDocument> => {
    try {
      // Use skip_sync=true to ensure we compute positions based on stored distances,
      // not from old visual positions that might be out of sync
      const response = await fetch('/api/compute-pylink-trajectory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ pylink_data: doc, n_steps: 12, skip_sync: true })
      })

      if (!response.ok) return doc
      const result = await response.json()

      if (result.status === 'success' && result.trajectories) {
        // Update meta.joints with first frame positions from simulation
        const updatedMetaJoints = { ...doc.meta.joints }
        for (const [jointName, positions] of Object.entries(result.trajectories)) {
          const posArray = positions as [number, number][]
          if (posArray && posArray.length > 0 && posArray[0]) {
            const [x, y] = posArray[0]
            if (updatedMetaJoints[jointName]) {
              updatedMetaJoints[jointName] = { ...updatedMetaJoints[jointName], x, y }
            } else {
              updatedMetaJoints[jointName] = { x, y, color: '#ff7f0e', zlevel: 0, show_path: true }
            }
          }
        }
        return { ...doc, meta: { ...doc.meta, joints: updatedMetaJoints } }
      }
    } catch (e) {
      console.warn('Could not sync positions with distances:', e)
    }
    return doc
  }

  // Load pylink graph from server (most recent)
  const loadPylinkGraph = async () => {
    try {
      showStatus('Loading...', 'action')
      const response = await fetch('/api/load-pylink-graph')

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const result = await response.json()

      if (result.status === 'success' && result.data) {
        // Repair any Revolute joints with incorrect parent references
        let repairedDoc = repairRevoluteJointReferences(result.data as PylinkDocument)
        // Sync positions with distances (ensures visual matches geometry)
        repairedDoc = await syncPositionsWithDistances(repairedDoc)
        setPylinkDoc(repairedDoc)
        setSelectedJoints([])
        setSelectedLinks([])
        clearTrajectory()  // Clear any old trajectory
        triggerMechanismChange()
        showStatus(`Loaded ${result.filename}`, 'success', 3000)
      } else {
        showStatus(result.message || 'No graphs to load', 'warning', 3000)
      }
    } catch (error) {
      showStatus(`Load error: ${error}`, 'error', 3000)
    }
  }

  return (
    <Box
      ref={containerRef}
      sx={{
        position: 'relative',
        width: '100%',
        maxWidth: '100vw',
        height: 'calc(100vh - 140px)',
        minWidth: CANVAS_MIN_WIDTH_PX,
        minHeight: CANVAS_MIN_HEIGHT_PX,
        pb: '38px', // Space for fixed footer (reduced from 44px)
        overflow: 'hidden'
      }}
    >
      {/* Canvas area */}
      <Paper
        ref={canvasRef}
        sx={{
          width: '100%',
          height: '100%',
          overflow: 'hidden',
          backgroundColor: canvasBgColor === 'default'
            ? 'var(--color-canvas-bg)'
            : canvasBgColor === 'white' ? '#ffffff'
            : canvasBgColor === 'cream' ? '#FAF3E1'
            : '#1a1a1a',
          border: '1px solid var(--color-border)',
          borderRadius: 2,
          position: 'relative',
          cursor: getCursorStyle()
        }}
      >
        <svg
          width="100%"
          height="100%"
          style={{ display: 'block' }}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          onMouseLeave={handleCanvasMouseUp}
          onClick={handleCanvasClick}
          onDoubleClick={handleCanvasDoubleClick}
        >
          {/* SVG Filter Definitions for Glow Effects */}
          <defs>
            {/* Joint type glows - uses joint's own color */}
            <filter id="glow-static" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#E74C3C" floodOpacity="0.8"/>
            </filter>
            <filter id="glow-crank" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#F39C12" floodOpacity="0.8"/>
            </filter>
            <filter id="glow-pivot" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#2196F3" floodOpacity="0.8"/>
            </filter>

            {/* Graph color glows for links/objects */}
            <filter id="glow-blue" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#1F77B4" floodOpacity="0.8"/>
            </filter>
            <filter id="glow-orange" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#FF7F0E" floodOpacity="0.8"/>
            </filter>
            <filter id="glow-green" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#2CA02C" floodOpacity="0.8"/>
            </filter>
            <filter id="glow-red" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#D62728" floodOpacity="0.8"/>
            </filter>
            <filter id="glow-purple" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#9467BD" floodOpacity="0.8"/>
            </filter>
            <filter id="glow-brown" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#8C564B" floodOpacity="0.8"/>
            </filter>
            <filter id="glow-pink" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#E377C2" floodOpacity="0.8"/>
            </filter>
            <filter id="glow-gray" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#7F7F7F" floodOpacity="0.8"/>
            </filter>
            <filter id="glow-olive" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#BCBD22" floodOpacity="0.8"/>
            </filter>
            <filter id="glow-cyan" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#17BECF" floodOpacity="0.8"/>
            </filter>

            {/* Move group glow (grey) - special case */}
            <filter id="glow-movegroup" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="2" floodColor="#9E9E9E" floodOpacity="0.6"/>
            </filter>
            {/* Merge highlight glow (cyan) - special case */}
            <filter id="glow-merge" x="-50%" y="-50%" width="200%" height="200%">
              <feDropShadow dx="0" dy="0" stdDeviation="4" floodColor="#00BCD4" floodOpacity="0.9"/>
            </filter>
          </defs>

          {/* Grid (toggleable via Settings) */}
          {showGrid && renderGrid()}

          {/* Completed drawn objects (polygons, shapes) - render BEFORE links so links are on top */}
          {renderDrawnObjects()}

          {/* Links - rendered on top of polygons so they can be clicked even inside polygons */}
          {renderLinks()}

          {/* Preview line during link creation */}
          {renderPreviewLine()}

          {/* Polygon preview during drawing */}
          {renderPolygonPreview()}

          {/* Target paths for trajectory optimization */}
          {renderTargetPaths()}

          {/* Target path preview during drawing */}
          {renderPathPreview()}

          {/* Trajectory dots for simulation results */}
          {renderTrajectories()}

          {/* Joints */}
          {renderJoints()}

          {/* Group selection box */}
          {renderSelectionBox()}

          {/* Measurement markers */}
          {renderMeasurementMarkers()}
          {renderMeasurementLine()}
        </svg>

        {/* ToolbarToggleButtonsContainer - Horizontal button bar for toggling toolbars */}
        {/* Contains: Tools, Links, Nodes, More buttons in a horizontal row */}
        <ToolbarToggleButtons
          openToolbars={openToolbars}
          onToggleToolbar={handleToggleToolbar}
        />

        {/* Draggable floating toolbars */}
        {Array.from(openToolbars).map(toolbarId => {
          const config = TOOLBAR_CONFIGS.find(c => c.id === toolbarId)
          if (!config) return null
          const dimensions = getToolbarDimensions(toolbarId)
          return (
            <DraggableToolbar
              key={toolbarId}
              id={toolbarId}
              title={config.title}
              icon={config.icon}
              initialPosition={getToolbarPosition(toolbarId)}
              onClose={() => handleToggleToolbar(toolbarId)}
              onPositionChange={handleToolbarPositionChange}
              minWidth={dimensions.minWidth}
              maxHeight={dimensions.maxHeight}
            >
              {renderToolbarContent(toolbarId)}
            </DraggableToolbar>
          )
        })}
      </Paper>

      {/* Fixed Footer Toolbar */}
      <FooterToolbar
        toolMode={toolMode}
        jointCount={pylinkDoc.pylinkage.joints.length}
        linkCount={Object.keys(pylinkDoc.meta.links).length}
        selectedJoints={selectedJoints}
        selectedLinks={selectedLinks}
        statusMessage={statusMessage}
        linkCreationState={linkCreationState}
        polygonDrawState={polygonDrawState}
        measureState={measureState}
        groupSelectionState={groupSelectionState}
        mergePolygonState={mergePolygonState}
        pathDrawState={pathDrawState}
        canvasWidth={canvasDimensions.width}
        onCancelAction={cancelAction}
        darkMode={darkMode}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteConfirmDialog.open}
        onClose={() => setDeleteConfirmDialog({ open: false, joints: [], links: [] })}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            confirmDelete()
          }
        }}
      >
        <DialogTitle sx={{ pb: 1 }}>
          Confirm Delete
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete {deleteConfirmDialog.joints.length + deleteConfirmDialog.links.length} items?
          </DialogContentText>
          <Box sx={{ mt: 2 }}>
            {deleteConfirmDialog.joints.length > 0 && (
              <Typography variant="body2" sx={{ mb: 0.5 }}>
                <strong>Joints:</strong> {deleteConfirmDialog.joints.join(', ')}
              </Typography>
            )}
            {deleteConfirmDialog.links.length > 0 && (
              <Typography variant="body2">
                <strong>Links:</strong> {deleteConfirmDialog.links.join(', ')}
              </Typography>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => setDeleteConfirmDialog({ open: false, joints: [], links: [] })}
            color="inherit"
          >
            Cancel
          </Button>
          <Button
            onClick={confirmDelete}
            color="error"
            variant="contained"
            autoFocus
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Joint Edit Modal */}
      <JointEditModal
        open={editingJointData !== null}
        onClose={() => setEditingJointData(null)}
        jointData={editingJointData}
        jointTypes={JOINT_TYPES}
        onRename={renameJoint}
        onTypeChange={(jointName, newType) => updateJointProperty(jointName, 'type', newType)}
        onShowPathChange={(jointName, showPath) => {
          setPylinkDoc(prev => ({
            ...prev,
            meta: {
              ...prev.meta,
              joints: {
                ...prev.meta.joints,
                [jointName]: {
                  ...prev.meta.joints[jointName],
                  show_path: showPath
                }
              }
            }
          }))
          // Update modal data
          setEditingJointData(prev => prev ? { ...prev, showPath } : null)
          // Trigger re-render of trajectories (paths are already computed, just need to re-render)
          // No need to re-simulate - just triggering state update refreshes the render
        }}
        darkMode={darkMode}
      />

      {/* Link Edit Modal */}
      <LinkEditModal
        open={editingLinkData !== null}
        onClose={() => setEditingLinkData(null)}
        linkData={editingLinkData}
        onRename={renameLink}
        onColorChange={(linkName, color) => {
          updateLinkProperty(linkName, 'color', color)
          // Update modal data
          setEditingLinkData(prev => prev ? { ...prev, color } : null)
        }}
        onGroundChange={(linkName, isGround) => {
          updateLinkProperty(linkName, 'isGround', isGround)
          // Update modal data
          setEditingLinkData(prev => prev ? { ...prev, isGround } : null)
        }}
        darkMode={darkMode}
      />
    </Box>
  )
}

export default PylinkBuilderTab
