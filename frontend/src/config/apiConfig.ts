export interface AppConfig {
  apiHost: string;
  apiPort: number;
  useRelativeApi: boolean;
}

export interface ApiSettings {
  apiHost: string;
  apiPort: number;
  useRelativeApi: boolean;
}

const STORAGE_KEY = "sa-aqg-api-settings";

export const DEFAULT_API_HOST = "192.168.1.198";
export const DEFAULT_API_PORT = 1408;

const DEFAULT_CONFIG: AppConfig = {
  apiHost: DEFAULT_API_HOST,
  apiPort: DEFAULT_API_PORT,
  useRelativeApi: false,
};

let cachedFileConfig: AppConfig | null = null;

function readLocalOverride(): Partial<ApiSettings> | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as Partial<ApiSettings>;
  } catch {
    return null;
  }
}

export function saveApiSettings(settings: ApiSettings): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
}

export function clearApiSettingsOverride(): void {
  localStorage.removeItem(STORAGE_KEY);
}

export async function loadAppConfig(): Promise<AppConfig> {
  if (cachedFileConfig) return cachedFileConfig;

  let fileConfig = { ...DEFAULT_CONFIG };
  try {
    const res = await fetch("/app-config.json", { cache: "no-store" });
    if (res.ok) {
      const json = (await res.json()) as Partial<AppConfig>;
      fileConfig = { ...DEFAULT_CONFIG, ...json };
    }
  } catch {
    console.warn("[api-config] Could not load /app-config.json; using defaults.");
  }

  cachedFileConfig = fileConfig;
  return fileConfig;
}

export async function getApiBaseUrl(): Promise<string> {
  const fileConfig = await loadAppConfig();
  const override = readLocalOverride();

  const useRelative =
    override?.useRelativeApi ?? fileConfig.useRelativeApi;
  if (useRelative) {
    return "/api";
  }

  const host = override?.apiHost ?? fileConfig.apiHost ?? DEFAULT_API_HOST;
  const port = override?.apiPort ?? fileConfig.apiPort ?? DEFAULT_API_PORT;
  return `http://${host}:${port}/api`;
}

export async function getResolvedApiSettings(): Promise<ApiSettings> {
  const fileConfig = await loadAppConfig();
  const override = readLocalOverride();
  return {
    apiHost: override?.apiHost ?? fileConfig.apiHost,
    apiPort: override?.apiPort ?? fileConfig.apiPort,
    useRelativeApi: override?.useRelativeApi ?? fileConfig.useRelativeApi,
  };
}
