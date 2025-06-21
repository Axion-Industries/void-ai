// API utility to connect frontend chat to backend AI
export async function sendChatMessage(prompt: string, options?: { max_new_tokens?: number; temperature?: number }) {
    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            prompt,
            max_new_tokens: options?.max_new_tokens ?? 100,
            temperature: options?.temperature ?? 0.8,
        }),
    });
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Unknown error");
    }
    const data = await response.json();
    return data.text;
}
