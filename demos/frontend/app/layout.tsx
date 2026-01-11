import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  weight: ["400", "500", "600", "700"],
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
    <html lang="en" className="dark" style={{ fontFamily: 'var(--font-inter)' }}>
      <body
        className={`${inter.variable} antialiased`}
        style={{ 
          fontFamily: 'var(--font-inter), sans-serif',
          WebkitFontSmoothing: 'antialiased',
          MozOsxFontSmoothing: 'grayscale'
        }}
      >
        {children}
        {/* Portal container for modal */}
        <div id="modal-portal" />
      </body>
    </html>
  );
}
