import Link from "next/link";
export default function NotFound() {
  return (
    <section className="panel empty">
      <h1>View not found</h1>
      <Link className="button primary" href="/">
        Return to overview
      </Link>
    </section>
  );
}
