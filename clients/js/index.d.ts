export interface ConfabClaim {
  text: string;
  confidence: number;
  level: 'high' | 'medium' | 'low';
}

export interface ConfabResult {
  content: string;
  claims: ConfabClaim[];
  confidence: number;
  raw: any;
}

export interface ConfabOptions {
  baseUrl?: string;
  model?: string;
}

export interface ChatOptions {
  model?: string;
  temperature?: number;
  [key: string]: any;
}

export interface ConfabClient {
  chat(prompt: string, options?: ChatOptions): Promise<ConfabResult>;
  chatStream(prompt: string, options?: ChatOptions): AsyncGenerator<any>;
}

export function confab(options?: ConfabOptions): ConfabClient;
