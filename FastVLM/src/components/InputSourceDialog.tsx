import { useState, useCallback } from "react";
import GlassContainer from "./GlassContainer";
import GlassButton from "./GlassButton";
import { GLASS_EFFECTS } from "../constants";

const ERROR_TYPES = {
  HTTPS: "https",
  NOT_SUPPORTED: "not-supported",
  PERMISSION: "permission",
  GENERAL: "general",
} as const;

const VIDEO_CONSTRAINTS = {
  video: {
    width: { ideal: 1920, max: 1920 },
    height: { ideal: 1080, max: 1080 },
    facingMode: "user",
  },
};

const SCREEN_CONSTRAINTS = {
  video: {
    width: { ideal: 1920, max: 1920 },
    height: { ideal: 1080, max: 1080 },
  },
  audio: false,
};

interface ErrorInfo {
  type: (typeof ERROR_TYPES)[keyof typeof ERROR_TYPES];
  message: string;
}

interface InputSourceDialogProps {
  onSourceSelected: (stream: MediaStream, sourceType: 'webcam' | 'screen' | 'file') => void;
}

type InputSource = 'webcam' | 'screen' | 'file';

export default function InputSourceDialog({ onSourceSelected }: InputSourceDialogProps) {
  const [selectedSource, setSelectedSource] = useState<InputSource | null>(null);
  const [isRequesting, setIsRequesting] = useState(false);
  const [error, setError] = useState<ErrorInfo | null>(null);

  const getErrorInfo = (err: unknown): ErrorInfo => {
    if (!navigator.mediaDevices) {
      return {
        type: ERROR_TYPES.HTTPS,
        message: "Media access requires a secure connection (HTTPS)",
      };
    }

    if (err instanceof DOMException) {
      switch (err.name) {
        case "NotAllowedError":
          return {
            type: ERROR_TYPES.PERMISSION,
            message: "Media access denied",
          };
        case "NotFoundError":
          return {
            type: ERROR_TYPES.GENERAL,
            message: "No camera found",
          };
        case "NotReadableError":
          return {
            type: ERROR_TYPES.GENERAL,
            message: "Camera is in use by another application",
          };
        case "OverconstrainedError":
          return {
            type: ERROR_TYPES.GENERAL,
            message: "Camera doesn't meet requirements",
          };
        case "SecurityError":
          return {
            type: ERROR_TYPES.HTTPS,
            message: "Security error accessing media",
          };
        default:
          return {
            type: ERROR_TYPES.GENERAL,
            message: `Media error: ${err.name}`,
          };
      }
    }

    return {
      type: ERROR_TYPES.GENERAL,
      message: "Failed to access media",
    };
  };

  const requestWebcamAccess = useCallback(async () => {
    setIsRequesting(true);
    setError(null);

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("NOT_SUPPORTED");
      }

      const stream = await navigator.mediaDevices.getUserMedia(VIDEO_CONSTRAINTS);
      onSourceSelected(stream, 'webcam');
    } catch (err) {
      const errorInfo = getErrorInfo(err);
      setError(errorInfo);
      console.error("Error accessing webcam:", err, errorInfo);
    } finally {
      setIsRequesting(false);
    }
  }, [onSourceSelected]);

  const requestScreenAccess = useCallback(async () => {
    setIsRequesting(true);
    setError(null);

    try {
      if (!navigator.mediaDevices?.getDisplayMedia) {
        throw new Error("Screen sharing not supported");
      }

      const stream = await navigator.mediaDevices.getDisplayMedia(SCREEN_CONSTRAINTS);
      onSourceSelected(stream, 'screen');
    } catch (err) {
      const errorInfo = getErrorInfo(err);
      setError(errorInfo);
      console.error("Error accessing screen:", err, errorInfo);
    } finally {
      setIsRequesting(false);
    }
  }, [onSourceSelected]);

  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Create a video element that will be used directly instead of canvas stream
    const videoUrl = URL.createObjectURL(file);

    // Create a mock stream that signals this is a file source
    const canvas = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    const mockStream = canvas.captureStream(1);

    // Store the video file URL on the stream for later use
    (mockStream as any).videoFileUrl = videoUrl;

    onSourceSelected(mockStream, 'file');
  }, [onSourceSelected]);

  const renderIcon = (source: InputSource) => {
    const iconClass = "w-8 h-8";

    switch (source) {
      case 'webcam':
        return (
          <svg className={`${iconClass} text-blue-400`} fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
          </svg>
        );
      case 'screen':
        return (
          <svg className={`${iconClass} text-green-400`} fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 011 1v8a1 1 0 01-1 1H4a1 1 0 01-1-1V4zm2 1v6h10V5H5z" clipRule="evenodd" />
            <path d="M10 15a1 1 0 011-1h2a1 1 0 110 2h-2a1 1 0 01-1-1zM7 14a1 1 0 100 2h2a1 1 0 100-2H7z" />
          </svg>
        );
      case 'file':
        return (
          <svg className={`${iconClass} text-purple-400`} fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 011 1v1a1 1 0 01-1 1H4a1 1 0 01-1-1v-1zM3 7a1 1 0 011-1h12a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1V7zM4 9h12v2H4V9z" clipRule="evenodd" />
          </svg>
        );
    }
  };

  if (selectedSource && isRequesting) {
    return (
      <div className="absolute inset-0 text-white flex items-center justify-center p-8">
        <GlassContainer className="rounded-3xl shadow-2xl">
          <div className="p-8 text-center space-y-6">
            <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent mx-auto" />
            <h2 className="text-2xl font-bold text-gray-100">
              {selectedSource === 'webcam' ? 'Requesting Camera Access' :
               selectedSource === 'screen' ? 'Requesting Screen Access' :
               'Loading Video File'}
            </h2>
            <p className="text-gray-400">Please allow access in your browser to continue...</p>
          </div>
        </GlassContainer>
      </div>
    );
  }

  return (
    <div className="absolute inset-0 text-white flex items-center justify-center p-8">
      <div className="max-w-2xl w-full space-y-6">
        <GlassContainer className="rounded-3xl shadow-2xl">
          <div className="p-8 text-center space-y-6">
            <h2 className="text-3xl font-bold text-gray-100">Choose Input Source</h2>
            <p className="text-gray-400">Select how you want to provide video for captioning</p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
              {/* Webcam Option */}
              <GlassContainer
                className="rounded-2xl p-6 cursor-pointer hover:scale-105 transition-transform duration-200"
                bgColor={GLASS_EFFECTS.COLORS.DEFAULT_BG}
                onClick={() => {
                  setSelectedSource('webcam');
                  requestWebcamAccess();
                }}
              >
                <div className="flex flex-col items-center space-y-3">
                  <div className="w-16 h-16 rounded-full bg-blue-500/20 flex items-center justify-center">
                    {renderIcon('webcam')}
                  </div>
                  <h3 className="text-lg font-semibold text-gray-200">Webcam</h3>
                  <p className="text-sm text-gray-400 text-center">Use your camera for live captioning</p>
                </div>
              </GlassContainer>

              {/* Screen Recording Option */}
              <GlassContainer
                className="rounded-2xl p-6 cursor-pointer hover:scale-105 transition-transform duration-200"
                bgColor={GLASS_EFFECTS.COLORS.DEFAULT_BG}
                onClick={() => {
                  setSelectedSource('screen');
                  requestScreenAccess();
                }}
              >
                <div className="flex flex-col items-center space-y-3">
                  <div className="w-16 h-16 rounded-full bg-green-500/20 flex items-center justify-center">
                    {renderIcon('screen')}
                  </div>
                  <h3 className="text-lg font-semibold text-gray-200">Screen</h3>
                  <p className="text-sm text-gray-400 text-center">Record and caption your screen</p>
                </div>
              </GlassContainer>

              {/* Video File Option */}
              <GlassContainer
                className="rounded-2xl p-6 cursor-pointer hover:scale-105 transition-transform duration-200"
                bgColor={GLASS_EFFECTS.COLORS.DEFAULT_BG}
              >
                <label className="flex flex-col items-center space-y-3 cursor-pointer">
                  <div className="w-16 h-16 rounded-full bg-purple-500/20 flex items-center justify-center">
                    {renderIcon('file')}
                  </div>
                  <h3 className="text-lg font-semibold text-gray-200">Video File</h3>
                  <p className="text-sm text-gray-400 text-center">Upload a video file to caption</p>
                  <input
                    type="file"
                    accept="video/*"
                    className="hidden"
                    onChange={handleFileSelect}
                  />
                </label>
              </GlassContainer>
            </div>
          </div>
        </GlassContainer>

        {/* Error Display */}
        {error && (
          <GlassContainer
            className="rounded-2xl shadow-2xl"
            bgColor={GLASS_EFFECTS.COLORS.ERROR_BG}
          >
            <div className="p-6 text-center">
              <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center mx-auto mb-4">
                <svg className="w-8 h-8 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <h3 className="text-xl font-bold text-gray-100 mb-2">Access Failed</h3>
              <p className="text-red-400 mb-4">{error.message}</p>

              <GlassButton
                onClick={() => {
                  setError(null);
                  setSelectedSource(null);
                }}
                className="px-6 py-3"
              >
                Try Different Source
              </GlassButton>
            </div>
          </GlassContainer>
        )}
      </div>
    </div>
  );
}
