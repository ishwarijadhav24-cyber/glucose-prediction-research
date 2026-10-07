"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity } from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";
import { cn } from "@/lib/utils";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/method", label: "Method" },
  { href: "/research/loso", label: "Research", match: "/research" },
  { href: "/lab", label: "Forecast Lab" },
];

export function Navbar() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background/90 backdrop-blur supports-[backdrop-filter]:bg-background/70">
      <div className="container flex h-14 max-w-screen-xl items-center justify-between gap-4">
        <Link href="/" className="flex items-center gap-2 shrink-0">
          <span className="grid place-items-center w-8 h-8 rounded-lg bg-primary/10">
            <Activity className="h-4.5 w-4.5 text-primary" />
          </span>
          <span className="font-semibold hidden sm:inline">GlucoCast <span className="text-muted-foreground font-normal">· 30-min forecast</span></span>
        </Link>
        <nav className="flex items-center gap-1 overflow-x-auto text-sm font-medium">
          {LINKS.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.match ?? l.href);
            return (
              <Link key={l.href} href={l.href}
                className={cn("px-3 py-1.5 rounded-md whitespace-nowrap transition-colors",
                  active ? "bg-primary/10 text-primary" : "text-foreground/60 hover:text-foreground")}>
                {l.label}
              </Link>
            );
          })}
        </nav>
        <ThemeToggle />
      </div>
    </header>
  );
}
