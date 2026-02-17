import { useState, useEffect, useCallback, useRef } from "react";
import GlassContainer from "./GlassContainer";
import GlassButton from "./GlassButton";
import { GLASS_EFFECTS } from "../constants";

interface VideoScrubberProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  isVisible: boolean;
}

export default function VideoScrubber({ videoRef, isVisible }: VideoScrubberProps) {
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const scrubberRef = useRef<HTMLInputElement>(null);

  const formatTime = useCallback((seconds: number) => {
    if (!isFinite(seconds) || isNaN(seconds)) {
      return "0:00";
    }
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);

  const updateProgress = useCallback(() => {
    const video = videoRef.current;
    if (video && !isDragging) {
      const time = isFinite(video.currentTime) ? video.currentTime : 0;
      const dur = isFinite(video.duration) ? video.duration : 0;
      setCurrentTime(time);
      setDuration(dur);
      setIsPlaying(!video.paused);
    }
  }, [videoRef, isDragging]);

  const togglePlayPause = useCallback(() => {
    const video = videoRef.current;
    if (video) {
      if (video.paused) {
        video.play();
      } else {
        video.pause();
      }
    }
  }, [videoRef]);

  const handleSeek = useCallback((newTime: number) => {
    const video = videoRef.current;
    if (video && !isNaN(newTime)) {
      video.currentTime = newTime;
      setCurrentTime(newTime);
    }
  }, [videoRef]);

  const handleScrubberChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const newTime = parseFloat(event.target.value);
    handleSeek(newTime);
  }, [handleSeek]);

  const handleMouseDown = useCallback(() => {
    setIsDragging(true);
  }, []);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      const dur = isFinite(video.duration) ? video.duration : 0;
      const time = isFinite(video.currentTime) ? video.currentTime : 0;
      setDuration(dur);
      setCurrentTime(time);
    };

    const handleTimeUpdate = () => {
      updateProgress();
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);

    // Update immediately if metadata is already loaded
    if (video.readyState >= 1) {
      handleLoadedMetadata();
    }

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
    };
  }, [videoRef, updateProgress]);

  useEffect(() => {
    if (isDragging) {
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, handleMouseUp]);

  if (!isVisible) {
    return null;
  }

  const progressPercentage = duration > 0 && isFinite(duration) && isFinite(currentTime) 
    ? Math.min((currentTime / duration) * 100, 100) 
    : 0;

  return (
    <div 
      className={`absolute bottom-4 left-4 right-4 z-[200] transition-opacity duration-300 ${
        isHovered || isDragging ? 'opacity-100' : 'opacity-90'
      }`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{ pointerEvents: 'auto' }}
    >
      <GlassContainer
        bgColor={GLASS_EFFECTS.COLORS.DEFAULT_BG}
        className="rounded-lg px-4 py-3 shadow-lg"
      >
        <div className="flex items-center space-x-4">
          {/* Play/Pause Button */}
          <GlassButton
            onClick={togglePlayPause}
            className="w-10 h-10 rounded-full flex items-center justify-center p-0"
            aria-label={isPlaying ? "Pause video" : "Play video"}
          >
            {isPlaying ? (
              <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M6 4a1 1 0 011 1v10a1 1 0 11-2 0V5a1 1 0 011-1zM14 4a1 1 0 011 1v10a1 1 0 11-2 0V5a1 1 0 011-1z" clipRule="evenodd" />
              </svg>
            ) : (
              <svg className="w-5 h-5 text-white ml-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M6.267 3.455a.5.5 0 01.531-.024L15.5 8.5a.5.5 0 010 .872l-8.702 5.069a.5.5 0 01-.765-.436V3.455z" clipRule="evenodd" />
              </svg>
            )}
          </GlassButton>

          {/* Current Time */}
          <div className="text-white text-sm font-mono min-w-[3.5rem]">
            {formatTime(currentTime)}
          </div>

          {/* Timeline Scrubber */}
          <div className="flex-1 relative px-2">
            <input
              ref={scrubberRef}
              type="range"
              min={0}
              max={duration || 100}
              step={0.1}
              value={currentTime}
              onChange={handleScrubberChange}
              onMouseDown={handleMouseDown}
              className="w-full h-1.5 bg-transparent rounded-lg appearance-none cursor-pointer
                         [&::-webkit-slider-track]:bg-gray-600/50
                         [&::-webkit-slider-track]:rounded-lg
                         [&::-webkit-slider-track]:h-1.5
                         [&::-webkit-slider-thumb]:appearance-none 
                         [&::-webkit-slider-thumb]:w-4 
                         [&::-webkit-slider-thumb]:h-4 
                         [&::-webkit-slider-thumb]:rounded-full 
                         [&::-webkit-slider-thumb]:bg-white 
                         [&::-webkit-slider-thumb]:shadow-lg
                         [&::-webkit-slider-thumb]:cursor-pointer
                         [&::-webkit-slider-thumb]:border-2
                         [&::-webkit-slider-thumb]:border-blue-500
                         [&::-webkit-slider-thumb]:transition-transform
                         [&::-moz-range-track]:bg-gray-600/50
                         [&::-moz-range-track]:rounded-lg
                         [&::-moz-range-track]:h-1.5
                         [&::-moz-range-track]:border-0
                         [&::-moz-range-thumb]:w-4 
                         [&::-moz-range-thumb]:h-4 
                         [&::-moz-range-thumb]:rounded-full 
                         [&::-moz-range-thumb]:bg-white 
                         [&::-moz-range-thumb]:border-2
                         [&::-moz-range-thumb]:border-blue-500
                         [&::-moz-range-thumb]:cursor-pointer
                         [&::-moz-range-thumb]:transition-transform
                         hover:[&::-webkit-slider-thumb]:scale-110
                         hover:[&::-moz-range-thumb]:scale-110"
              style={{
                background: `linear-gradient(to right, 
                  #3b82f6 0%, 
                  #3b82f6 ${progressPercentage}%, 
                  rgba(75, 85, 99, 0.3) ${progressPercentage}%, 
                  rgba(75, 85, 99, 0.3) 100%)`
              }}
            />
          </div>

          {/* Duration */}
          <div className="text-white text-sm font-mono min-w-[3.5rem] text-right">
            {formatTime(duration)}
          </div>
        </div>
      </GlassContainer>
    </div>
  );
}