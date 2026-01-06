import type { Metadata } from "next";
import { Space_Grotesk } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-space-grotesk",
  weight: ["300", "400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "EXAIM - Clinical Decision Support System",
  description: "AI-powered clinical decision support with real-time agent reasoning traces",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" style={{ fontFamily: 'var(--font-space-grotesk)' }}>
      <body
        className={`${spaceGrotesk.variable} antialiased`}
        style={{ fontFamily: 'var(--font-space-grotesk), sans-serif' }}
      >
        {children}
        {/* Portal container for modal */}
        <div id="modal-portal" />
      </body>
    </html>
  );
}
