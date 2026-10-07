"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Users, UserPlus, Globe, Microscope } from "lucide-react";
import { cn } from "@/lib/utils";

const TABS = [
  { href: "/research/loso", label: "Objective 1 · Unseen patients", short: "Objective 1", icon: Users },
  { href: "/research/personalization", label: "Objective 2 · Personalization", short: "Objective 2", icon: UserPlus },
  { href: "/research/external", label: "Objective 3 · New dataset", short: "Objective 3", icon: Globe },
  { href: "/research/analyses", label: "Deeper analyses & limits", short: "Analyses", icon: Microscope },
];

export default function ResearchLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  return (
    <div className="flex-1">
      <div className="border-b bg-muted/30 sticky top-14 z-30 backdrop-blur">
        <div className="container max-w-screen-xl flex gap-1 overflow-x-auto py-2">
          {TABS.map((t) => (
            <Link key={t.href} href={t.href}
              className={cn("flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium whitespace-nowrap transition-colors",
                pathname === t.href ? "bg-background shadow-sm text-primary border" : "text-muted-foreground hover:text-foreground")}>
              <t.icon className="w-4 h-4" />
              <span className="hidden md:inline">{t.label}</span><span className="md:hidden">{t.short}</span>
            </Link>
          ))}
        </div>
      </div>
      <div className="container max-w-screen-xl py-10 space-y-12">{children}</div>
    </div>
  );
}
