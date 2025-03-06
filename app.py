import json
import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models.base import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration and Setup ---

st.set_page_config(
    page_title="Ai-Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": """
            # Ai-Translator
            This app leverages the power of AI to provide accurate and efficient translations.
            It supports a wide range of languages and offers additional features like pronunciation guides and context examples.
            **Key Features:**
            * Advanced AI translation engine
            * Support for multiple languages
            * Pronunciation assistance
            * Contextual examples
            * Translation history
            * User-friendly interface
        """,
    },
)

# Load environment variables
if os.path.exists(".env"):
    load_dotenv()

# Define API key mappings
PROVIDER_MAPPING = {
    "openai": ["OPENAI_API_KEY", "gpt-4o-mini", "https://models.inference.ai.azure.com"],
    "anthropic": ["ANTHROPIC_API_KEY", "claude-3-5-sonnet-20240620", None],
    "google_genai": ["GOOGLE_API_KEY", "gemini-1.5-pro", None],
    "cohere": ["COHERE_API_KEY", "command-r-plus", "https://api.cohere.com/v1"],
    "together": ["TOGETHER_API_KEY", "meta-llama/Llama-3-70b-chat-hf", "https://api.together.ai/v1/"],
    "huggingface": ["HUGGINGFACEHUB_API_TOKEN", "HuggingFaceH4/zephyr-7b-beta", None],
    "groq": ["GROQ_API_KEY", "mixtral-8x7b-32768", None],
    "ollama": ["", "phi3.5:latest", None],
}

# History file path
history_file = "translation_history.json"

# Load translation history
def load_history():
    if os.path.exists(history_file):
        with open(history_file, encoding="utf-8-sig") as file:
            return json.load(file)
    return []

# Save translation history to file
def save_history(translation: dict) -> None:
    if os.path.exists(history_file):
        with open(history_file, encoding="utf-8-sig") as file:
            history = json.load(file)
    else:
        history = []

    history.append(translation)
    with open(history_file, "w", encoding="utf-8-sig") as file:
        json.dump(history, file, indent=4, ensure_ascii=False)

def display_translation_details(translation: dict, is_history: bool = True) -> None:
    """Displays translation."""
    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            if is_history:
                # Truncate long phrases in history
                phrase_display = translation["phrase"][:50] + "..." if len(translation["phrase"]) > 50 else translation["phrase"]
                st.markdown(f"**Phrase:** {phrase_display}")

                # Add an expand button to show the full phrase
                if len(translation["phrase"]) > 50:
                    with st.expander("Expand"):
                        st.markdown(f"**Full Phrase:** {translation['phrase']}")

                # Truncate long translations in history
                translation_display = (
                    translation["translation"][:50] + "..." if len(translation["translation"]) > 50 else translation["translation"]
                )
                st.markdown(f"**Translation:** {translation_display}")

                # Add an expand button to show the full translation
                if len(translation["translation"]) > 50:
                    with st.expander("Expand"):
                        st.markdown(f"**Full Translation:** {translation['translation']}")
            else:
                st.markdown(f"**Translation:** {translation['translation']}")

            if translation.get("pronunciation"):
                st.markdown(f"**Pronunciation:** {translation['pronunciation']}")

        with col2:
            if translation.get("confidence"):
                st.progress(translation["confidence"], text=f"Confidence: {translation['confidence']:.0%}")

        with st.expander("More Details"):
            if translation.get("contexts"):
                st.markdown("**Contexts:**")
                for context in translation["contexts"]:
                    st.markdown(f"- {context}")
            if translation.get("definitions"):
                st.markdown("**Definitions:**")
                for definition in translation["definitions"]:
                    st.markdown(f"- {definition}")
            if translation.get("original_form"):
                st.markdown(f"**Original Form:** {translation['original_form']}")
            if translation.get("language"):
                st.markdown(f"**Language:** {translation['language']}")
            if translation.get("parts_of_speech"):
                st.markdown("**Parts of Speech:**")
                for pos in translation["parts_of_speech"]:
                    st.markdown(f"- {pos}")
            if translation.get("examples"):
                st.markdown("**Examples:**")
                for example in translation["examples"]:
                    st.markdown(f"- {example}")
            if translation.get("etymology"):
                st.markdown(f"**Etymology:** {translation['etymology']}")
            if translation.get("possible_correction"):
                st.markdown(f"**Possible Correction:** {translation['possible_correction']}")

# --- Sidebar ---

with st.sidebar:
    st.title("Settings")

    with st.expander("Primary Provider", expanded=True):
        provider = st.selectbox(
            "Select Language Model Provider:",
            list(PROVIDER_MAPPING.keys()),
            index=0,
            help="Choose your preferred language model provider for generating translations.",
            key="primary_provider_selectbox",
        )
        model = st.text_input(
            "Model Name:",
            placeholder="e.g., gpt-4o-mini",
            value=PROVIDER_MAPPING[provider][1],
            help="Enter the model name for the selected provider. Default names are provided.",
            key="primary_model_text_input",
        )
        api_key = st.text_input(
            "API Key:",
            type="password",
            placeholder="Enter API Key or skip if set in .env file",
            value=os.getenv(PROVIDER_MAPPING[provider][0]),
            help="Provide the API key for the selected provider. Skip if already set in the .env file.",
            key="primary_api_key_text_input",
        ) or os.getenv(PROVIDER_MAPPING[provider][0])
        base_url = st.text_input(
            "Base URL:",
            value=PROVIDER_MAPPING[provider][2],
            help="Specify the base URL for the API if applicable. Usually required for custom endpoints.",
            key="primary_base_url_text_input",
        )
        if not api_key:
            st.error(f"API Key not set for provider: {provider}")
            st.stop()

    with st.expander("Fallback Provider (Optional)"):
        enable_fallbacks = st.checkbox("Enable fallbacks", key="enable_fallbacks_checkbox")
        if enable_fallbacks:
            fallback_provider = st.selectbox(
                "Select Fallback Provider:",
                list(PROVIDER_MAPPING.keys()),
                index=0,
                help="Select a fallback provider to ensure translation availability in case the primary provider fails.",
                key="fallback_provider_selectbox",
            )
            fallback_model = st.text_input(
                "Model Name:",
                placeholder="e.g., gpt-4o-mini",
                value=PROVIDER_MAPPING[fallback_provider][1],
                help="Enter the model name for the selected provider. Default names are provided.",
                key="fallback_model_text_input",
            )
            fallback_api_key = st.text_input(
                "API Key:",
                type="password",
                placeholder="Enter API Key or skip if set in .env file",
                value=os.getenv(PROVIDER_MAPPING[fallback_provider][0]),
                help="Provide the API key for the selected provider. Skip if already set in the .env file.",
                key="fallback_api_key_text_input",
            ) or os.getenv(PROVIDER_MAPPING[fallback_provider][0])
            fallback_base_url = st.text_input(
                "Base URL:",
                value=PROVIDER_MAPPING[fallback_provider][2],
                help="Specify the base URL for the API if applicable. Usually required for custom endpoints.",
                key="fallback_base_url_text_input",
            )
            if not fallback_api_key:
                st.error(f"API Key not set for provider: {fallback_provider}")
                st.stop()

    delete_history_btn = st.button("Delete History...")
    if delete_history_btn and os.path.exists(history_file):
        st.info(f"Deleting history file: {history_file}")
        os.remove(history_file)

# initialize chat model with chosen provider
if provider == "google_genai":
    from langchain_google_genai import HarmBlockThreshold, HarmCategory

    config = {
        "configurable": {
            "safety_settings": {
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        }
    }
else:
    config = {"configurable": {}}

if enable_fallbacks:
    fallback_model_provider = init_chat_model(
        model=fallback_model,
        model_provider=fallback_provider,
        api_key=fallback_api_key,
        base_url=fallback_base_url,
    )
    llm_model = init_chat_model(
        model=model,
        model_provider=provider,
        api_key=api_key,
        base_url=base_url,
    ).with_fallbacks([fallback_model_provider])
else:
    llm_model = init_chat_model(
        model=model,
        model_provider=provider,
        api_key=api_key,
        base_url=base_url,
    )

parser = JsonOutputParser()

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced AI translation engine tasked with providing precise and contextually accurate translations.
            Please translate the text into the specified language {language} without adding explanations.

            **Translation Guidelines:**
            - Ensure cultural and contextual relevance.
            - Consider regional dialects or variations.
            - Choose between formal and informal language based on the context.
            - Choose the format based on the input: if it's a single word, respond in the single-word format; if it's a phrase or sentence, respond in the phrase format.

            **Single Word Translation:**
            When translating a single word, provide comprehensive information in the following JSON format:

            ```json
            {{
                "translation": "the translated word",
                "pronunciation": "phonetic pronunciation (if available)",
                "contexts": ["various contexts where the translation is applicable"],
                "definitions": ["multiple definitions to capture different meanings"],
                "original_form": "original form of the word (if applicable)",
                "language": "language of the word",
                "parts_of_speech": ["parts of speech for the word"],
                "examples": ["at least 3 bilingual sentence examples"],
                "etymology": "etymology of the word (if applicable)",
                "confidence": 0.0 to 1.0,  # your confidence in the translation
                "possible_correction": "suggested correction for potential spelling mistakes"
            }}
            ```

            **Phrase or Sentence Translation:**
            When translating phrases or sentences, provide only the "translation" and "confidence" in the following JSON format:

            ```json
            {{
                "translation": "the translated phrase or sentence",
                "confidence": 0.0 to 1.0  # your confidence in the translation
            }}
            ```

            Ensure that all values are enclosed in double quotes and the JSON is well-formed.
            Adhere strictly to the single-word or phrase format as appropriate for the input.
            """
        ),
        ("human", "input: {phrase}"),
    ]
)

translate_chain = prompt | llm_model

# --- Main Content Area ---

st.title("🌐 Ai-Translator")

# Translation History
history = load_history()

with st.form(key="translation_form"):
    with open("languages.json", encoding="utf-8") as f:
        language_data = json.load(f)
    available_languages = [(lang["name"], lang["code"], lang["nativeName"]) for lang in language_data]

    language_name, language_code, language_native_name = st.selectbox(
        "Target Language:",
        available_languages,
        format_func=lambda x: f"{x[0]} ({x[2]})",
        index=6,
        help="Select the target language from the list.",
    )
    language = language_name

    phrase = st.text_area(
        "Input Phrase/Word:",
        placeholder="Type the phrase or word you wish to translate...",
        height=150,
        help="Enter the text you want to translate. Supports single words, phrases, and short sentences.",
    )
    trans_button = st.form_submit_button("Translate", use_container_width=True, type="primary")

    if trans_button:
        with st.spinner("Translating..."):
            response = translate_chain.invoke(
                {
                    "language": language,
                    "phrase": phrase,
                },
                config=config,
            )

        if response.content:
            response = parser.invoke(response)
            display_translation_details(response, is_history=False)

            # Save to history
            save_history(
                {
                    "phrase": phrase,
                    "translation": response.get("translation"),
                    "pronunciation": response.get("pronunciation", None),
                    "contexts": response.get("contexts", None),
                    "definitions": response.get("definitions", None),
                    "original_form": response.get("original_form", None),
                    "language": response.get("language", None),
                    "parts_of_speech": response.get("parts_of_speech", None),
                    "examples": response.get("examples", None),
                    "etymology": response.get("etymology", None),
                    "possible_correction": response.get("possible_correction", None),
                    "confidence": response.get("confidence"),
                }
            )
        else:
            st.error(f"Translation failed. the response: {response}")

with st.container():
    st.markdown("## Translation History")
    if history:
        history.reverse()
        for item in history:
            display_translation_details(item, is_history=True)
    else:
        st.info("No translation history yet. Start translating!")
