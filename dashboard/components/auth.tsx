"use client";

import { createContext, useContext, useEffect, useState } from "react";
import {
  createUserWithEmailAndPassword, onIdTokenChanged, sendEmailVerification,
  sendPasswordResetEmail, signInWithEmailAndPassword, signOut, User,
} from "firebase/auth";
import Link from "next/link";
import { firebaseAuth, firebaseConfigured } from "@/lib/firebase";

type Session = { user: User | null; mode: string; loading: boolean; error: string };
const SessionContext = createContext<Session>({ user: null, mode: "", loading: true, error: "" });
export const useSession = () => useContext(SessionContext);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<Session>({ user: null, mode: "", loading: true, error: "" });
  useEffect(() => {
    let active = true;
    let unsubscribe = () => {};
    fetch("/api/auth/config", { cache: "no-store" }).then(async (response) => {
      if (!response.ok) throw new Error("Authentication service is unavailable.");
      const config = await response.json();
      if (!active) return;
      if (!["local", "firebase"].includes(config.mode)) throw new Error("Invalid authentication configuration.");
      if (config.mode === "local") {
        setState({ user: null, mode: "local", loading: false, error: "" });
        return;
      }
      if (!firebaseConfigured) throw new Error("Firebase login needs the dashboard environment configuration. See docs/FIREBASE_SETUP.md.");
      const auth = firebaseAuth()!;
      unsubscribe = onIdTokenChanged(auth, (user) => {
        if (active) setState({ user, mode: "firebase", loading: false, error: "" });
      }, () => {
        if (active) setState({ user: null, mode: "firebase", loading: false, error: "Session could not be restored. Reload to retry." });
      });
    }).catch((issue) => {
      if (active) setState({ user: null, mode: "firebase", loading: false, error: issue.message });
    });
    return () => { active = false; unsubscribe(); };
  }, []);
  return <SessionContext.Provider value={state}>{children}</SessionContext.Provider>;
}

export function AuthGate({ children }: { children: React.ReactNode }) {
  const session = useSession();
  if (session.loading) return <main className="auth-shell"><p role="status">Restoring your session…</p></main>;
  if (session.error) return <main className="auth-shell"><h1>Workspace setup</h1><p role="alert">{session.error}</p><button onClick={() => window.location.reload()}>Retry connection</button></main>;
  if (session.mode === "firebase" && (!session.user || !session.user.emailVerified)) return <LoginForm />;
  // Changing identity remounts all workspace state so another user's results cannot linger.
  return <div key={session.user?.uid || "local"}>{children}</div>;
}

export function AccountButton() {
  const { user, mode } = useSession();
  if (mode === "local") return <Link href="/login" className="muted">Local research mode</Link>;
  return <><span className="muted">{user?.email}</span><button className="button secondary" onClick={async () => {
    await signOut(firebaseAuth()!);
  }}>Sign out</button></>;
}

export function LoginForm() {
  const { user, mode } = useSession();
  const [signup, setSignup] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  async function perform(action: () => Promise<unknown>, success: string) {
    setBusy(true); setMessage("");
    try { await action(); setMessage(success); }
    catch { setMessage("Could not complete this request. Check your details, connection and Firebase provider settings, then retry."); }
    finally { setBusy(false); }
  }
  if (mode === "local") return <main className="auth-shell"><h1>Firebase is not enabled yet</h1><p>The workspace is running in local research mode. No user authentication or cloud history is active. Follow docs/FIREBASE_SETUP.md when your Firebase configuration is available.</p><Link href="/">Open local workspace</Link></main>;
  if (user?.emailVerified) return <main className="auth-shell"><h1>You are signed in</h1><p>{user.email}</p><Link href="/">Open workspace</Link></main>;
  if (user) return <main className="auth-shell"><h1>Verify your email</h1><p>A verification link was requested for {user.email}. Verify your address before accessing saved research.</p><button disabled={busy} onClick={() => void perform(async () => {
    await user.reload(); await user.getIdToken(true); window.location.reload();
  }, "")}>I have verified my email</button><button disabled={busy} onClick={() => void perform(() => sendEmailVerification(user), "Verification email requested.")}>Resend verification email</button><button onClick={() => void signOut(firebaseAuth()!)}>Use another account</button><p role="status">{message}</p></main>;
  return <main className="auth-shell"><p className="eyebrow">PHARMALENS / RESEARCH WORKSPACE</p><h1>{signup ? "Create your account" : "Sign in"}</h1><p>Access your predictions, explanations and saved batch results.</p><form onSubmit={(event) => {
    event.preventDefault();
    void perform(async () => {
      const auth = firebaseAuth()!;
      if (signup) {
        const result = await createUserWithEmailAndPassword(auth, email.trim(), password);
        await sendEmailVerification(result.user);
      } else await signInWithEmailAndPassword(auth, email.trim(), password);
      setPassword("");
    }, signup ? "Check your email to verify your account." : "Signed in.");
  }}><label>Email<input type="email" required autoComplete="email" maxLength={254} value={email} onChange={(event) => setEmail(event.target.value)} /></label><label>Password<input type="password" required minLength={signup ? 12 : 1} maxLength={128} autoComplete={signup ? "new-password" : "current-password"} value={password} onChange={(event) => setPassword(event.target.value)} /></label>{signup ? <p className="muted">Use at least 12 characters. Firebase enforces your project's configured password policy.</p> : null}<button disabled={busy} type="submit">{busy ? "Please wait…" : signup ? "Create account" : "Sign in"}</button></form><button disabled={busy} onClick={() => { setSignup(!signup); setMessage(""); setPassword(""); }}>{signup ? "Already have an account? Sign in" : "Create an account"}</button><button disabled={busy || !email.trim()} onClick={() => void perform(() => sendPasswordResetEmail(firebaseAuth()!, email.trim()), "If an account exists, a password reset email has been requested.")}>Reset password</button><p role="status" aria-live="polite">{message}</p></main>;
}
