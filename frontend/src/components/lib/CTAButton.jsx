export default function CTAButton({ children, onClick, className = "" }) {
  return (
    <button onClick={onClick} className={`rounded-lg px-4 py-2 bg-blue-500 text-white ${className}`}>
      {children}
    </button>
  );
}
