import React, { useState } from 'react'
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Alert,
  CircularProgress,
  Paper
} from '@mui/material'
import { StatusResponse } from '../response_models'
import cheetahGif from '../assets/cheetah_run.gif'
import acinonyxLogo from '../assets/acinonyx_logo.png'

const StatusAboutTab: React.FC = () => {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const callStatusEndpoint = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/status')

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setStatus(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box sx={{
      py: 3,
      px: 2,
      maxWidth: 900,
      margin: '0 auto'
    }}>
      {/* Hero Section with Cheetah Animation */}
      <Paper
        elevation={0}
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          py: 4,
          px: 3,
          mb: 3,
          backgroundColor: 'var(--color-bg-tan)',
          borderRadius: 3,
          border: '1px solid var(--color-border-light)',
          transition: 'all 0.25s ease'
        }}
      >
        <img
          src={cheetahGif}
          alt="Acinonyx - The Cheetah"
          style={{
            maxWidth: '280px',
            width: '100%',
            height: 'auto',
            marginBottom: '16px'
          }}
        />
        <Typography
          variant="h4"
          sx={{
            fontWeight: 700,
            color: 'var(--color-text-primary)',
            textAlign: 'center',
            mb: 1
          }}
        >
          Acinonyx
        </Typography>
        <Typography
          variant="body1"
          sx={{
            color: 'var(--color-text-secondary)',
            textAlign: 'center',
            fontStyle: 'italic'
          }}
        >
          High-Speed Mechanical Linkage Simulation
        </Typography>
      </Paper>

      {/* Backend Status Card */}
      <Card sx={{
        mb: 3,
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--color-border-light)',
        transition: 'all 0.25s ease'
      }}>
        <CardContent>
          <Typography
            variant="h6"
            gutterBottom
            sx={{
              fontWeight: 600,
              color: 'var(--color-text-primary)'
            }}
          >
            Backend Status
          </Typography>

          <Box sx={{ mb: 2 }}>
            <Button
              variant="contained"
              onClick={callStatusEndpoint}
              disabled={loading}
              sx={{
                mr: 2,
                backgroundColor: 'var(--color-primary)',
                '&:hover': {
                  backgroundColor: '#e67300'
                }
              }}
            >
              {loading ? <CircularProgress size={20} color="inherit" /> : 'Check Status'}
            </Button>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              Error: {error}
            </Alert>
          )}

          {status && (
            <Alert severity="success">
              <Typography variant="body1">
                <strong>Status:</strong> {status.status}
              </Typography>
              {status.message && (
                <Typography variant="body2">
                  <strong>Message:</strong> {status.message}
                </Typography>
              )}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* About Card */}
      <Card sx={{
        mb: 3,
        backgroundColor: 'var(--color-surface)',
        border: '1px solid var(--color-border-light)',
        transition: 'all 0.25s ease'
      }}>
        <CardContent>
          <Typography
            variant="h6"
            gutterBottom
            sx={{
              fontWeight: 600,
              color: 'var(--color-text-primary)'
            }}
          >
            About Acinonyx
          </Typography>
          <Typography
            variant="body2"
            paragraph
            sx={{ color: 'var(--color-text-secondary)' }}
          >
            Acinonyx is a mechanical linkage design and simulation platform. Designed in particular to aid the design and simulation of organic automata with complex constraints.
          </Typography>
          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              mb: 1,
              color: 'var(--color-text-primary)'
            }}
          >
            Features:
          </Typography>
          <Typography
            variant="body2"
            component="ul"
            sx={{
              pl: 2,
              color: 'var(--color-text-secondary)',
              '& li': { mb: 0.5 }
            }}
          >
            <li>Interactive mechanical linkage simulation</li>
            <li>Graph-based system design</li>
            <li>Real-time visualization and animation</li>
            <li>FastAPI backend for efficient processing</li>
          </Typography>
        </CardContent>
      </Card>

      {/* Footer with Logo */}
      <Box sx={{
        mt: 4,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 2
      }}>
        <img
          src={acinonyxLogo}
          alt="Acinonyx"
          style={{
            width: '48px',
            height: '48px',
            objectFit: 'contain',
            borderRadius: '8px',
            opacity: 0.7
          }}
        />
        <Typography
          variant="caption"
          align="center"
          display="block"
          sx={{
            color: 'var(--color-text-muted)',
            fontSize: '0.75rem'
          }}
        >
          Attribution-ShareAlike 4.0 License
          <br />
          <a
            href="https://github.com/bastidas/automata"
            target="_blank"
            rel="noopener noreferrer"
            style={{
              color: 'var(--color-primary)',
              textDecoration: 'none'
            }}
          >
            github.com/bastidas/automata
          </a>
        </Typography>
      </Box>
    </Box>
  )
}

export default StatusAboutTab
