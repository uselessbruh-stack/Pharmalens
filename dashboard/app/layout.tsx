import type { Metadata } from "next";
import { Workspace } from "@/components/workspace";
import { AuthGate, AuthProvider } from "@/components/auth";
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
        <AuthProvider><AuthGate><Workspace>{children}</Workspace></AuthGate></AuthProvider>
      </body>
    </html>
  );
}
