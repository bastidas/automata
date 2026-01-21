/**
 * More Toolbar - File Operations, Demos, Validation
 *
 * Note: Animation controls have been moved to AnimateToolbar
 */
import React from 'react'
import { Box, Typography, Button, Tooltip, Divider } from '@mui/material'

export interface MoreToolbarProps {
  // Demo operations
  loadDemo4Bar: () => void
  loadDemoLeg: () => void
  loadDemoWalker: () => void
  loadDemoComplex: () => void

  // File operations
  loadPylinkGraphLast: () => void
  loadFromFile: () => void
  savePylinkGraph: () => void
  savePylinkGraphAs: () => void
  validateMechanism: () => void
}

export const MoreToolbar: React.FC<MoreToolbarProps> = ({
  loadDemo4Bar,
  loadDemoLeg,
  loadDemoWalker,
  loadDemoComplex,
  loadPylinkGraphLast,
  loadFromFile,
  savePylinkGraph,
  savePylinkGraphAs,
  validateMechanism,
}) => {
  return (
    <Box sx={{ p: 1.5 }}>
      {/* ═══════════════════════════════════════════════════════════════════════
          FILE OPERATIONS (first - most used)
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        File Operations
      </Typography>
      <Box sx={{ display: 'flex', gap: 0.5, mt: 1, mb: 1 }}>
        <Button
          variant="outlined"
          size="small"
          onClick={loadFromFile}
          sx={{ flex: 1, textTransform: 'none', fontSize: '0.7rem' }}
        >
          ↑ Load
        </Button>
        <Button
          variant="outlined"
          size="small"
          onClick={loadPylinkGraphLast}
          sx={{ flex: 1, textTransform: 'none', fontSize: '0.7rem' }}
        >
          ↑ Last
        </Button>
      </Box>
      <Box sx={{ display: 'flex', gap: 0.5, mb: 1.5 }}>
        <Button
          variant="outlined"
          size="small"
          onClick={savePylinkGraph}
          sx={{ flex: 1, textTransform: 'none', fontSize: '0.7rem' }}
        >
          ↓ Save
        </Button>
        <Button
          variant="outlined"
          size="small"
          onClick={savePylinkGraphAs}
          sx={{ flex: 1, textTransform: 'none', fontSize: '0.7rem' }}
        >
          ↓ Save As
        </Button>
      </Box>

      <Divider sx={{ my: 1 }} />

      {/* ═══════════════════════════════════════════════════════════════════════
          DEMOS
          ═══════════════════════════════════════════════════════════════════════ */}
      <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary' }}>
        Demos
      </Typography>
      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, mt: 1, mb: 1.5 }}>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={loadDemo4Bar}
          sx={{ textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
        >
          ◇ Four Bar
        </Button>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={loadDemoLeg}
          sx={{ textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
        >
          ◇ Leg
        </Button>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={loadDemoWalker}
          sx={{ textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
        >
          ◇ Walker
        </Button>
        <Button
          variant="outlined"
          fullWidth
          size="small"
          onClick={loadDemoComplex}
          sx={{ textTransform: 'none', justifyContent: 'flex-start', fontSize: '0.75rem' }}
        >
          ◇ Complex
        </Button>
      </Box>

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

export default MoreToolbar
