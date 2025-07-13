const API_BASE_URL = (import.meta as any).env.DEV 
  ? 'http://127.0.0.1:5000' 
  : 'https://sayhey-chatbot.onrender.com';

export interface ChatRequest {
  message: string;
  user_id: string;
}

export interface ChatResponse {
  response: string;
  session_cancelled?: boolean;
  warning_count?: number;
}

export const sendMessage = async (request: ChatRequest): Promise<ChatResponse> => {
  const response = await fetch(`${API_BASE_URL}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request)
  });

  if (response.status === 403) {
    // Handle ban/session cancellation
    const data = await response.json();
    return data;
  }

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return response.json();
}; 