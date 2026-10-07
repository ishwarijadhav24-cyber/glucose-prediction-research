import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { ThemeProvider } from "@/components/theme-provider";
import { ReactQueryProvider } from "@/components/providers";
import { Disclaimer, Footer } from "@/components/layout/disclaimer";
import { Navbar } from "@/components/layout/navbar";
import { InteractiveBackground } from "@/components/interactive-background";
import { NavigationProgress } from "@/components/navigation-progress";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "GlucoCast — 30-minute glucose forecasting research",
  description:
    "Subject-independent, causal 30-minute blood glucose forecasting for Type 1 diabetes: LOSO evaluation, personalization and external transfer to HUPA-UCM.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased min-h-screen flex flex-col`}>
        <ThemeProvider defaultTheme="system" enableSystem>
          <ReactQueryProvider>
            <InteractiveBackground />
            <NavigationProgress />
            <div className="relative z-10 flex-1 flex flex-col">
              <Disclaimer />
              <Navbar />
              <main className="flex-1 flex flex-col">{children}</main>
              <Footer />
            </div>
          </ReactQueryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
