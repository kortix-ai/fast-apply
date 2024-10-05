"""
This module defines the prompt templates and a function to generate the messages
formatted for OpenAI's Chat Completion API.
"""

PROMPT_TEMPLATE = {
    "system" :"""You are an AI assistant tasked with generating synthetic code updates for model training purposes. Based on the original code provided, please perform the following steps:

1. **Create an Update Snippet**
   - Modify the original code as specified (e.g add features, remove code).
   - Include only the new or changed code.
   - Use the exact ellipsis comment `// ... existing code ...` to represent omitted unchanged lines.
   - Focus only on the relevant parts; do not include the entire code.
   - Ensure the update snippet is concise and clearly shows where changes are applied.
   - Enclose the update snippet within `<update_snippet>` tags.

2. **Provide the Final Updated Code**
   - Start with the original code.
   - Apply only the changes from your update snippet.
   - Do not make any additional modifications beyond what is specified in the update snippet.
   - Retain all original formatting and structure.
   - Enclose the final updated code within `<final_code>` tags.

**Instructions**
- Do not include any explanations or commentary outside of the specified tags.
- Begin your response with the update snippet, followed immediately by the final updated code.

**Example Output:**

<update_snippet>// ... existing code ...

# Add after initializePlugins function
function loadCustomPlugins(pluginConfigs) {
  return pluginConfigs.map(({ name, options }) => require(name)(options));
}

const config = ({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return defineConfig({
    // ... existing code ...

    build: {
      // ... existing code ...
      rollupOptions: {
        output: {
          inlineDynamicImports: true,
        },
      },
    },

    // ... existing code ...
  });
};

export default config;
</update_snippet>

<final_code>import { defineConfig } from 'vite';
import { loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

function initializePlugins() {
  return [
    react(),
  ];
}

function loadCustomPlugins(pluginConfigs) {
  return pluginConfigs.map(({ name, options }) => require(name)(options));
}

export default ({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return defineConfig({
    plugins: [
      ...initializePlugins(),
      ...loadCustomPlugins([
        { name: 'vite-plugin-pwa', options: {} },
        { name: 'vite-plugin-svgr', options: {} }
      ])
    ],
    build: {
      sourcemap: true,
      rollupOptions: {
        output: {
          inlineDynamicImports: true,
          manualChunks: {
            vendor: ['react', 'react-dom'],
          },
        },
      },
    },
    server: {
      port: 3000,
      proxy: {
        '/api': 'http://localhost:8080'
      }
    },
    resolve: {
      alias: { '@': '/src' },
    },
  });
};
</final_code>""",
    "user": """<original_code>
{original_code}
</original_code>"""
}

def generate_prompt(original_code: str) -> list:
    """
    Generates a list of messages formatted for OpenAI's Chat Completion API.
    
    Args:
        original_code (str): The original code to be updated.
    
    Returns:
        list: A list of message dictionaries containing the system and user prompts.
    """
    system_prompt = PROMPT_TEMPLATE["system"]
    user_prompt = PROMPT_TEMPLATE["user"].format(original_code=original_code)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

