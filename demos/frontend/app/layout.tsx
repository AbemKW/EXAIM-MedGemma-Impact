import type { Metadata } from "next";
import { Inter } from "next/font/google";
import localFont from "next/font/local";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  weight: ["400", "500", "600", "700"],
});

const powerGrotesk = localFont({
  src: [
    {
      path: "./fonts/PowerGrotesk-Regular.ttf",
      weight: "400",
      style: "normal",
    },
  ],
  variable: "--font-power-grotesk",
  fallback: ["system-ui", "sans-serif"],
});

export const metadata: Metadata = {
  title: "EXAID - Clinical Decision Support System",
  description: "AI-powered clinical decision support with real-time agent reasoning traces",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" style={{ fontFamily: 'var(--font-power-grotesk)' }}>
      <body
        className={`${inter.variable} ${powerGrotesk.variable} antialiased`}
        style={{ fontFamily: 'var(--font-power-grotesk), sans-serif' }}
      >
        {children}
        {/* Portal container for modal */}
        <div id="modal-portal" />
      </body>
    </html>
  );
}
