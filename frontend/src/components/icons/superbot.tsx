export function SuperBotLogoSVG({
  width = 32,
  height = 32,
  className,
}: {
  width?: number;
  height?: number;
  className?: string;
}) {
  return (
    <svg
      width={width}
      height={height}
      viewBox="0 0 48 48"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-label="SuperBot logo"
    >
      <rect
        x="6"
        y="12"
        width="36"
        height="28"
        rx="8"
        fill="currentColor"
        opacity="0.12"
      />
      <rect
        x="6"
        y="12"
        width="36"
        height="28"
        rx="8"
        stroke="currentColor"
        strokeWidth="2.5"
      />
      <circle cx="17" cy="26" r="3.5" fill="currentColor" />
      <circle cx="31" cy="26" r="3.5" fill="currentColor" />
      <path
        d="M18 34c1.8 1.6 4 2.4 6 2.4s4.2-.8 6-2.4"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
      <path
        d="M24 12V7"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
      />
      <circle cx="24" cy="5" r="2.5" fill="currentColor" />
    </svg>
  );
}
