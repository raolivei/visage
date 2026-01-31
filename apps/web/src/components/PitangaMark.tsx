/**
 * Pitanga Logo
 * Shared branding component for Pitanga projects
 */

// Color palette
const LEAF = "#5a6e3a";
const LEAF_LIGHT = "#6b8044";
const BERRY = "#a82c24";

export const PitangaMark = ({ className = "w-5 h-5" }: { className?: string }) => (
  <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" className={className}>
    <path d="M16 11V14" stroke={LEAF} strokeWidth="0.75" strokeLinecap="round" />
    <path d="M16 11.5C16 11.5 15.2 7 12.5 7C9.8 7 9.5 9.5 9.5 10C9.5 10.5 9.8 13.5 13 13.5C15.2 13.5 16 11.5 16 11.5Z" fill={LEAF} />
    <path d="M16 11.5C16 11.5 16.8 7 19.5 7C22.2 7 22.5 9.5 22.5 10C22.5 10.5 22.2 13.5 19 13.5C16.8 13.5 16 11.5 16 11.5Z" fill={LEAF_LIGHT} fillOpacity="0.85" />
    <circle cx="11.5" cy="18.5" r="3.8" fill={BERRY} />
    <circle cx="20.5" cy="18.5" r="3.8" fill={BERRY} />
    <circle cx="16" cy="23.8" r="4.2" fill={BERRY} />
  </svg>
);

export default PitangaMark;
