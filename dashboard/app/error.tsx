"use client";
export default function ErrorPage({
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  return (
    <section className="panel empty">
      <h1>This view could not be loaded</h1>
      <p>Your dataset and model artifacts have not been modified.</p>
      <button className="button primary" onClick={reset}>
        Try again
      </button>
    </section>
  );
}
