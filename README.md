# Ai-Translator

**Harness the power of AI for accurate and efficient translations.**

This Streamlit application leverages advanced language models to provide high-quality translations across multiple languages. It goes beyond basic translation by offering features like pronunciation guides, context examples, and detailed word information.

## Features

* **Advanced AI Translation Engine:** Utilizes state-of-the-art language models (OpenAI, Anthropic, Google PaLM, etc.) for accurate translations.
* **Multi-Language Support:** Translate between a wide array of languages with ease.
* **Pronunciation Assistance:** Get phonetic pronunciations for translated words.
* **Contextual Examples:** See how translations are used in context for better understanding.
* **Word Details:**  For single words, explore in-depth information, including definitions, parts of speech, etymology, and more.
* **Translation History:** Keep track of your past translations for reference.
* **User-Friendly Interface:**  Intuitive and easy-to-use design for a seamless translation experience.
* **Customizable:** Select your preferred language model provider and configure settings.
* **Fallback Provider:**  Option to set up a fallback provider for uninterrupted translations.


## Installation

1. **Clone the repository:**

   ```bash
   git clone [Link of the repo]
   cd Ai-Translator
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys:**

   * Rename `.env-example` file into `.env` in the project root directory.
   * Obtain API keys for your chosen language model providers (e.g., OpenAI, Anthropic, Google).
   * Add the API keys to the `.env` file.

4. **Run the application:**

   ```bash
   streamlit run app.py
   ```


## Usage

1. **Select your language model provider and model** in the sidebar settings.
2. **Choose the target language** from the dropdown.
3. **Enter the phrase or word** you want to translate in the text area.
4. **Click "Translate"** to get your translation.
5. **Explore the translation details,** including pronunciation, confidence score, and more.
6. **Review your translation history** below the main translation area.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance this project.


## Acknowledgements

* Streamlit - for the amazing framework for building interactive web applications.
* Langchain - for simplifying the integration with language models.
