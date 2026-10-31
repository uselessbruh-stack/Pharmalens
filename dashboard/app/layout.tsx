import type { Metadata } from "next";
import { Workspace } from "@/components/workspace";
import "./globals.css";

export const metadata: Metadata = {
  title: "PharmaLens | Research workspace",
  description: "An auditable workspace for drug–target affinity research.",
};
export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <Workspace>{children}</Workspace>
      </body>
    </html>
  );
}
