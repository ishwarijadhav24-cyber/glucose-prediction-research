import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { ReactQueryProvider } from "@/components/providers";
import { Disclaimer } from "@/components/layout/disclaimer";
import { Navbar } from "@/components/layout/navbar";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Glucose Prediction Research",
  description: "AI-powered causal forecasting for continuous glucose monitoring.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased min-h-screen flex flex-col`}>
        <ThemeProvider
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <ReactQueryProvider>
            <Disclaimer />
            <Navbar />
            <main className="flex-1 flex flex-col">
              {children}
            </main>
          </ReactQueryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
