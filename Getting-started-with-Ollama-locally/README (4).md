
# Learning Ollama & Walkthrough 

# Ollama Integration Examples
## This repository demonstrates various ways to interact with Ollama's API, including direct API calls, OpenAI-compatible interface, and LangChain integration.
## Overview
### This code provides examples of:

1. Direct Ollama API usage
2. OpenAI-compatible interface
3. LangChain integration
4. Image processing with LLaVA
5. Vector embeddings and similarity search


### We Have two API\'s to consider natively by Ollama : 

#### 1. Generate {#1-generate}

#### 2. chat {#2-chat}

Reference:
<https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion>



``` python
import requests
import json

text = "Hello this is something about my home country, egypt "

response = requests.post('http://localhost:11434/api/generate', 
        json={
            "model": "llama3:8b",
            "prompt": f"Summarize this article in 3 bullet points: {text}",
            "stream": False,
            "options": {
                "temperature": 0.3,
            }}
            )
```



``` python
response.json()
```

 {.output .execute_result execution_count="2"}
    {'model': 'llama3:8b',
     'created_at': '2024-11-25T17:40:16.0643624Z',
     'response': "I'm happy to help! However, I don't see an article attached or provided. Could you please share the article with me so I can summarize it in 3 bullet points for you?",
     'done': True,
     'done_reason': 'stop',
     'context': [128006,
      882,
      128007,
      271,
      9370,
      5730,
      553,
      420,
      4652,
      304,
      220,
      18,
      17889,
      3585,
      25,
      22691,
      420,
      374,
      2555,
      922,
      856,
      2162,
      3224,
      11,
      384,
      13849,
      220,
      128009,
      128006,
      78191,
      128007,
      271,
      40,
      2846,
      6380,
      311,
      1520,
      0,
      4452,
      11,
      358,
      1541,
      956,
      1518,
      459,
      4652,
      12673,
      477,
      3984,
      13,
      16910,
      499,
      4587,
      4430,
      279,
      4652,
      449,
      757,
      779,
      358,
      649,
      63179,
      433,
      304,
      220,
      18,
      17889,
      3585,
      369,
      499,
      30],
     'total_duration': 11081718000,
     'load_duration': 8152036600,
     'prompt_eval_count': 33,
     'prompt_eval_duration': 424000000,
     'eval_count': 40,
     'eval_duration': 2500000000}




``` python
response.json()['response']
```

 {.output .execute_result execution_count="3"}
    "I'm happy to help! However, I don't see an article attached or provided. Could you please share the article with me so I can summarize it in 3 bullet points for you?"




``` python
from pprint import pprint as pprint
pprint(response.json()['response'])
```

 {.output .stream .stdout}
    ("I'm happy to help! However, I don't see an article attached or provided. "
     'Could you please share the article with me so I can summarize it in 3 bullet '
     'points for you?')



}
``` python
import requests
import json


response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3:8b",
        "messages": [ 
            {
                "role": "user",
                "content": "Hello! what are the beautiful things in egypt?"
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.3,
            "seed": 101, 
            "num_predict":100, # stop generating at 100 tokens only,
        }
    }
)
```


``` python
response.json()
```

 {.output .execute_result execution_count="47"}
    {'model': 'llama3:8b',
     'created_at': '2024-11-25T18:36:09.1719012Z',
     'message': {'role': 'assistant',
      'content': 'Egypt is a treasure trove of beauty, rich in history, culture, and natural wonders. Here are some of the most stunning and awe-inspiring attractions:\n\n1. **Pyramids of Giza**: The last remaining ancient wonder of the Seven Wonders of the World, these pyramids are an absolute must-see. The Great Pyramid of Khufu is the largest and most impressive.\n2. **Nile River**: The lifeblood of Egypt, the Nile stretches over 6,695'},
     'done_reason': 'length',
     'done': True,
     'total_duration': 9261657600,
     'load_duration': 2987260200,
     'prompt_eval_count': 21,
     'prompt_eval_duration': 429000000,
     'eval_count': 100,
     'eval_duration': 5842000000}



}
``` python
from pprint import pprint as pprint

pprint(response.json()['message']['content'])
```

 {.output .stream .stdout}
    ('Egypt is a treasure trove of beauty, rich in history, culture, and natural '
     'wonders. Here are some of the most stunning and awe-inspiring attractions:\n'
     '\n'
     '1. **Pyramids of Giza**: The last remaining ancient wonder of the Seven '
     'Wonders of the World, these pyramids are an absolute must-see. The Great '
     'Pyramid of Khufu is the largest and most impressive.\n'
     '2. **Nile River**: The lifeblood of Egypt, the Nile stretches over 6,695')


\`### num_ctx option -\> context window size

<https://github.com/ollama/ollama/issues/3644#issuecomment-2057646417>\`



``` python
import requests
import json

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3:8b",
        "messages": [
            {
                "role": "system",
                "content": """You are a detailed story writer for anime"""
            },
            {
                "role": "user", 
                "content": """Create a professional story character for anime"""}
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,   
            "seed": 101,
            "num_ctx" : 8000
    }
    }
)
```


}
``` python
response.json()
```

 {.output .execute_result execution_count="60"}
    {'model': 'llama3:8b',
     'created_at': '2024-11-25T18:58:02.9606481Z',
     'message': {'role': 'assistant',
      'content': 'Here is a professional story character for an anime:\n\n**Name:** Kaito Yamato ()\n\n**Age:** 17 (High School Student)\n\n**Appearance:** Kaito has short, spiky black hair and piercing blue eyes. He stands at around 5\'9" with a lean yet athletic build. He often wears his school uniform, but outside of school, he favors casual clothing like hoodies and jeans.\n\n**Personality:** Kaito is a free spirit who values independence and self-reliance. As the eldest child in a family of five siblings, he\'s learned to be resourceful and adaptable. He\'s fiercely loyal to those he cares about and will go to great lengths to protect them. Despite his tough exterior, Kaito has a soft spot for those in need and is always willing to lend a helping hand.\n\n**Background:** Born into a loving but chaotic household, Kaito grew up surrounded by siblings who often drove him crazy. His parents, though well-meaning, were often absent-minded and left the kids to fend for themselves. As a result, Kaito developed strong survival instincts and learned to rely on himself. He\'s always been fascinated with technology and mechanics, spending countless hours tinkering with gadgets and machinery.\n\n**Abilities:**\n\n1. **Technopathy:** Kaito has an uncanny ability to understand and repair complex electronic devices. He can hack into computer systems, decode encrypted files, and even create his own software.\n2. **Mechanical Genius:** With a knack for taking apart and reassembling machines, Kaito is a whiz with tools and machinery. He\'s always fixing broken appliances or modifying his friends\' gadgets to make them more efficient.\n3. **Enhanced Senses:** As a result of his intense focus on technology, Kaito has developed heightened senses that allow him to perceive the world in unique ways. He can see subtle vibrations in electronic signals, hear faint whispers from machinery, and even feel the pulse of energy emanating from devices.\n\n**Goals:**\n\n1. **Uncover the Truth:** Kaito is determined to uncover the truth behind a series of mysterious events plaguing his town. As he delves deeper into the mystery, he\'ll encounter powerful forces that threaten the very fabric of reality.\n2. **Protect His Friends:** With his friends\' lives on the line, Kaito will stop at nothing to keep them safe from harm. He\'ll use his skills and abilities to outwit foes, repair damaged equipment, and even create new gadgets to aid in their quest.\n\n**Motivations:**\n\n1. **Justice:** Kaito believes that justice is key to a harmonious society. As he uncovers the truth behind the mysterious events, he\'ll strive to bring those responsible to account.\n2. **Family:** Despite his tough exterior, Kaito\'s family is everything to him. He\'ll do whatever it takes to keep them safe and happy.\n\n**Conflict:**\n\n1. **Inner Turmoil:** As Kaito delves deeper into the mystery, he\'ll struggle with the weight of responsibility and the fear of losing those he cares about.\n2. **External Threats:** Powerful forces will stop at nothing to silence Kaito and his friends. He\'ll face off against ruthless villains, corrupt authorities, and even rival tech-savvy individuals who seek to outdo him.\n\n**Themes:**\n\n1. **Self-Discovery:** As Kaito navigates the complexities of his world, he\'ll learn to balance his independence with a sense of responsibility and teamwork.\n2. **Technology vs. Humanity:** The anime will explore the consequences of relying too heavily on technology, highlighting the importance of human connection and empathy in a rapidly changing world.\n\nThis character profile provides a solid foundation for an engaging anime story. Kaito\'s unique abilities, strong sense of justice, and personal motivations create a compelling narrative that explores themes relevant to modern audiences.'},
     'done_reason': 'stop',
     'done': True,
     'total_duration': 58540859500,
     'load_duration': 20232100,
     'prompt_eval_count': 30,
     'prompt_eval_duration': 518000000,
     'eval_count': 804,
     'eval_duration': 58000000000}



}
``` python
from pprint import pprint as pprint

pprint(response.json()['message']['content'])
```

 {.output .stream .stdout}
    "What is\n\nI'make"



}
``` python
import base64
from PIL import Image
import io

def encode_image_to_base64(image_path):
   with open(image_path, 'rb') as image_file:
       return base64.b64encode(image_file.read()).decode('utf-8')

def send_image_to_llava(image_path):
   base64_image = encode_image_to_base64(image_path)
   
   response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llava",
            "messages": [
                {
                    "role": "user", 
                    "content": "what is in this image?",
                    "images": [base64_image]
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }
   )
   
   return response.json()

result = send_image_to_llava("./download.jpeg")
print(result)
```

 {.output .stream .stdout}
    {'model': 'llava', 'created_at': '2024-11-25T19:02:56.3292543Z', 'message': {'role': 'assistant', 'content': " The image shows a bouquet of flowers with green leaves, set against a grassy background. There are three types of flowers: two purple ones that resemble lavender or dill, one yellow flower that could be a daisy or marigold, and some small pink flowers that might be wildflowers or baby's breath. The arrangement is placed on a patch of green grass, and there appears to be a plant with broad leaves in the background. The overall impression is that of a natural, garden-like setting. "}, 'done_reason': 'stop', 'done': True, 'total_duration': 26571851400, 'load_duration': 12942153100, 'prompt_eval_count': 592, 'prompt_eval_duration': 1509000000, 'eval_count': 111, 'eval_duration': 12117000000}



}
``` python
from pprint import pprint as pprint

pprint(result['message']['content'])
```

 {.output .stream .stdout}
    (' The image shows a bouquet of flowers with green leaves, set against a '
     'grassy background. There are three types of flowers: two purple ones that '
     'resemble lavender or dill, one yellow flower that could be a daisy or '
     "marigold, and some small pink flowers that might be wildflowers or baby's "
     'breath. The arrangement is placed on a patch of green grass, and there '
     'appears to be a plant with broad leaves in the background. The overall '
     'impression is that of a natural, garden-like setting. ')



# OPENAI

Refrences : <https://github.com/ollama/ollama/blob/main/docs/openai.md>


}
``` python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama_fs',
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'tell me a joke!',
        }
    ],
    model='llama3:8b',
)
```


}
``` python
print(type(chat_completion))
chat_completion
```

 {.output .stream .stdout}
    <class 'openai.types.chat.chat_completion.ChatCompletion'>


 {.output .execute_result execution_count="73"}
    ChatCompletion(id='chatcmpl-557', choices=[Choice(finish_reason='stop', index=0, logprobs=None, message=ChatCompletionMessage(content="Here's one:\n\nWhy couldn't the bicycle stand up by itself?\n\n(wait for it...)\n\nBecause it was two-tired!\n\nHope that brought a smile to your face!", refusal=None, role='assistant', audio=None, function_call=None, tool_calls=None))], created=1732561540, model='llama3:8b', object='chat.completion', service_tier=None, system_fingerprint='fp_ollama', usage=CompletionUsage(completion_tokens=35, prompt_tokens=15, total_tokens=50, completion_tokens_details=None, prompt_tokens_details=None))



}
``` python
print(chat_completion.choices[0].message.content)
```

 {.output .stream .stdout}
    Here's one:

    Why couldn't the bicycle stand up by itself?

    (wait for it...)

    Because it was two-tired!

    Hope that brought a smile to your face!




``` python
from openai import OpenAI

# Initialize the client
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama_fs',
)

def basic_chat():
    """Basic chat completion example"""
    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What are the three laws of robotics?"}
        ]
    )
    print("Basic Chat Response:", response.choices[0].message.content)

basic_chat()
```

 {.output .stream .stdout}
    Basic Chat Response: The Three Laws of Robotics were formulated by science fiction author Isaac Asimov as part of his robot series. The laws are designed to ensure that robots behave in a way that is safe and beneficial to humans. Here are the three laws:

    1. **A robot may not injure a human being or, through inaction, allow a human being to come to harm**. This law prioritizes the safety of human life above all else.
    2. **A robot must obey the orders given it by human beings except where such orders would conflict with the First Law**. This law establishes the importance of following human instructions and gives precedence to the well-being of humans over the operations of the robot itself.
    3. **A robot must protect its own existence as long as such protection does not conflict with the First or Second Law**. This law acknowledges that a robot, like any other self-aware being, has its own interests and requires some protection, but only to the extent that it doesn't compromise human safety or orders.

    These laws were designed to provide guidelines for robotics development and operation, and they have since become an influential concept in science fiction and science.



## Langchain

refrences :
<https://github.com/ollama/ollama/blob/main/docs/tutorials/langchainpy.md>


}
``` python
from langchain_community.llms import Ollama
ollama = Ollama(
    base_url='http://localhost:11434',
    model="llama3"
)
print(ollama.invoke("why is the sky blue"))
```

 {.output .stream .stdout}
    The sky appears blue to us because of a phenomenon called Rayleigh scattering, named after the British physicist Lord Rayleigh. He discovered that shorter wavelengths of light (like blue and violet) are scattered more than longer wavelengths (like red and orange) by tiny molecules of gases in the atmosphere.

    Here's what happens:

    1. **Sunlight enters Earth's atmosphere**: When sunlight from the sun enters our atmosphere, it contains all the colors of the visible spectrum, including red, orange, yellow, green, blue, indigo, and violet.
    2. **Molecules scatter light**: The tiny molecules of gases like nitrogen (N2) and oxygen (O2) in the atmosphere are much smaller than the wavelength of light. These molecules randomly collide with the light particles (photons), causing them to change direction.
    3. **Blue light is scattered more**: Shorter wavelengths, like blue and violet, have a higher frequency and are scattered more efficiently by these tiny molecules. This is because they interact more strongly with the molecules due to their smaller size.
    4. **Red light continues straight**: Longer wavelengths, like red and orange, have a lower frequency and travel in a relatively straight line, experiencing less scattering.
    5. **Our eyes perceive the result**: As we look at the sky, our eyes see the combined effect of all this scattered light. The blue and violet light that's been scattered in all directions reaches our eyes from many different angles, making the sky appear blue.

    In addition to Rayleigh scattering, there are other factors that can affect the color of the sky:

    * **Dust and water vapor**: Tiny particles in the atmosphere can absorb or scatter certain wavelengths of light, changing the apparent color of the sky.
    * **Clouds**: Clouds can reflect and scatter sunlight, making the sky appear white or gray.
    * **Atmospheric conditions**: Atmospheric conditions like pollution, humidity, and temperature can also influence the color of the sky.

    So, to summarize, the sky appears blue because of the scattering of shorter wavelengths (blue and violet) by tiny molecules in the atmosphere, combined with other factors that can affect the color.




``` python
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize Ollama
ollama = Ollama(
    base_url='http://localhost:11434',
    model="llama3.2" 
)

# 1. Basic Template
template = """
Question: {question}
Please provide a detailed answer.
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

chain = LLMChain(llm=ollama, prompt=prompt)
response = chain.invoke({"question": "What causes northern lights?"})

# uncomment for prints
# print(response.keys())
# print(type(response))
# print(response)
print("Basic Chain Response:", response['text'])
```

 {.output .stream .stdout}
    Basic Chain Response: The Northern Lights, also known as the Aurora Borealis, is a breathtaking natural phenomenon that occurs in the Earth's polar regions. It is caused by charged particles from the sun interacting with the Earth's magnetic field and atmosphere.

    Here's a detailed explanation of how it works:

    **Step 1: Solar Wind**

    The Northern Lights are triggered by solar winds, which are streams of charged particles emitted by the sun during solar flares or coronal mass ejections. These particles are mostly electrons and protons, which are accelerated to high speeds by the intense magnetic fields in the sun's corona.

    **Step 2: Magnetic Field**

    When the solar wind reaches the Earth's magnetic field, it is redirected towards the poles. The Earth's magnetic field acts as a shield, deflecting the charged particles from the direction of the sun and guiding them towards the polar regions.

    **Step 3: Atmospheric Interaction**

    As the solar wind particles collide with the Earth's atmosphere, they transfer their energy to the atoms and molecules in the air. This energy is released as light, which we see as the Northern Lights. The color of the Northern Lights depends on the altitude at which the collision occurs:

    * Green: Oxygen atoms at altitudes of around 100-200 km
    * Red: Oxygen atoms at altitudes of around 200-300 km
    * Blue and Violet: Nitrogen molecules at altitudes of around 100-200 km

    **Step 4: Auroral Formation**

    The Northern Lights are formed when the solar wind particles interact with the Earth's atmosphere in a specific way. The charged particles collide with the atmospheric atoms and molecules, exciting them and causing them to emit light. This process is known as excitation.

    As the excited atoms and molecules return to their ground state, they release energy in the form of light, which we see as the Northern Lights. The shape and color of the Northern Lights are determined by the density of the solar wind particles and the altitude at which the collision occurs.

    **Additional Factors**

    Several additional factors can influence the appearance and intensity of the Northern Lights:

    * Solar activity: Increased solar activity can lead to more frequent and intense Northern Lights displays.
    * Geomagnetic storms: These occur when the Earth's magnetic field is disturbed, causing increased solar wind particle flux.
    * Atmospheric conditions: Clear skies and low cloud cover are necessary for optimal viewing conditions.
    * Moon phase: A new moon or a thin crescent moon can enhance the visibility of the Northern Lights.

    In summary, the Northern Lights are caused by charged particles from the sun interacting with the Earth's magnetic field and atmosphere. The resulting collisions between the solar wind particles and atmospheric atoms and molecules lead to the emission of light, which we see as this breathtaking natural phenomenon.




``` python
from langchain.output_parsers import CommaSeparatedListOutputParser

# Initialize Ollama
ollama = Ollama(
    base_url='http://localhost:11434',
    model="llama3.2" 
)

# 2. Sequential Chain Example
# First chain generates topics, second chain explains them
topic_template = """Generate three interesting subtopics about: {subject}
Output only the subtopics, comma separated."""

explanation_template = """Provide a brief explanation of the following topic: {topic}"""



topic_chain = LLMChain(
    llm=ollama,
    prompt=PromptTemplate(
        input_variables=["subject"],
        template=topic_template
    ),
    output_parser=CommaSeparatedListOutputParser()
)

explanation_chain = LLMChain(
    llm=ollama,
    prompt=PromptTemplate(
        input_variables=["topic"],
        template=explanation_template
    )
)

# Run sequential chain
topics = topic_chain.invoke({"subject": "quantum computing"})
for topic in topics:
    explanation = explanation_chain.invoke({"topic": topic})
    print(f"\nTopic: {topic}")
    print(f"Explanation: {explanation['text']}")
```



``` python
from langchain.memory import ConversationBufferMemory

# 3. Chat with Memory Example
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

chat_template = """
Chat History: {chat_history}
Human: {human_input}
Assistant: Let me help you with that.
"""

chat_prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"],
    template=chat_template
)

chat_chain = LLMChain(
    llm=ollama,
    prompt=chat_prompt,
    memory=memory,
    verbose=True
)

# Simulate a conversation
responses = [
    chat_chain.invoke({"human_input": "What is machine learning?"}),
    chat_chain.invoke({"human_input": "Can you give me some examples of its applications?"}),
    chat_chain.invoke({"human_input": "What skills do I need to learn it?"})
]

# 4. Custom Output Parser Example
class ModelEvaluation:
    def parse(self, text):
        try:
            lines = text.strip().split('\n')
            return {
                'strengths': [s.strip() for s in lines if 'Strength:' in s],
                'weaknesses': [w.strip() for w in lines if 'Weakness:' in s]
            }
        except Exception as e:
            return {"error": str(e)}

eval_template = """
Analyze the following AI model and list its strengths and weaknesses.
Model: {model_name}

Format your response as:
Strength: [strength point]
Weakness: [weakness point]
"""

eval_chain = LLMChain(
    llm=ollama,
    prompt=PromptTemplate(
        input_variables=["model_name"],
        template=eval_template
    ),
    output_parser=ModelEvaluation()
)

evaluation = eval_chain.invoke({"model_name": "GPT-3"})
```



``` python
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://python.langchain.com/docs/integrations/document_loaders/web_base/")
data = loader.load()
```


}
``` python
pprint(data)
```

 {.output .stream .stdout}
    [Document(page_content='\n\n\n\n\nWebBaseLoader | \uf8ffü¶úÔ∏è\uf8ffüîó LangChain\n\n\n\n\n\n\nSkip to main contentIntegrationsAPI ReferenceMoreContributingPeopleError referenceLangSmithLangGraphLangChain HubLangChain JS/TSv0.3v0.3v0.2v0.1\uf8ffüí¨SearchProvidersAnthropicAWSGoogleHugging FaceMicrosoftOpenAIMoreProvidersAcreomActiveloop Deep LakeAerospikeAI21 LabsAimAINetworkAirbyteAirtableAlchemyAleph AlphaAlibaba CloudAnalyticDBAnnoyAnthropicAnyscaleApache Software FoundationApache DorisApifyAppleArangoDBArceeArcGISArgillaArizeArthurArxivAscendAskNewsAssemblyAIAstra DBAtlasAwaDBAWSAZLyricsBAAIBagelBagelDBBaichuanBaiduBananaBasetenBeamBeautiful SoupBibTeXBiliBiliBittensorBlackboardbookend.aiBoxBrave SearchBreebs (Open Knowledge)BrowserbaseBrowserlessByteDanceCassandraCerebrasCerebriumAIChaindeskChromaClarifaiClearMLClickHouseClickUpCloudflareClovaCnosDBCogniSwitchCohereCollege ConfidentialCometConfident AIConfluenceConneryContextCouchbaseCozeC TransformersCTranslate2CubeDappier AIDashVectorDatabricksDatadog TracingDatadog LogsDataForSEODataheraldDedocDeepInfraDeepSparseDiffbotDingoDBDiscordDocArrayDoctranDocugamiDocusaurusDriaDropboxDSPyDuckDBDuckDuckGo SearchE2BEden AIElasticsearchElevenLabsEmbedchainEpsillaEtherscanEverly AIEverNoteExaFacebook - MetaFalkorDBFaunaFiddlerFigmaFireCrawlFireworks AIFlyteForefront AIFriendli AIFriendli AIGeopandasGitGitBookGitHubGitLabGoldenGoogleSerper - Google Search APIGooseAIGPT4AllGradientGraphsignalGrobidGroqGutenbergHacker NewsHazy ResearchHeliconeHologresHTML to textHuaweiHugging FaceIBMIEIT SystemsiFixitiFlytekIMSDbInfinispan VSInfinityInfinoIntelIuguJaguarJavelin AI GatewayJina AIJohnsnowlabsJoplinKDB.AIKineticaKoboldAIKonkoKoNLPYK√πzuLabel StudiolakeFSLanceDBLangChain Decorators ‚ú®LanternLinkupLiteLLMLlamaIndexLlama.cppLlamaEdgellamafileLLMonitorLocalAILog10MariTalkMarqoMediaWikiDumpMeilisearchMemcachedMetalMicrosoftMilvusMindsDBMinimaxMistralAIMLflow AI Gateway for LLMsMLflowMLXModalModelScopeModern TreasuryMomentoMongoDBMongoDB AtlasMotherduckMot√∂rheadMyScaleNAVERNeo4jNLPCloudNomicNotion DBNucliaNVIDIAObsidianOracle Cloud Infrastructure (OCI)OctoAIOllamaOntotext GraphDBOpenAIOpenLLMOpenSearchOpenWeatherMapOracleAI Vector SearchOutlineOutlinesPandasPebbloPerplexityPetalsPostgres EmbeddingPGVectorPineconePipelineAIPortkeyPredibasePrediction GuardPremAIPromptLayerPsychicPubMedPygmalionAIQdrantRAGatouillerank_bm25Ray ServeRebuffRedditRedisRemembrallReplicateRoamSema4 (fka Robocorp)RocksetRunhouseRWKV-4Salute DevicesSAPScrapeGraph AISearchApiSearxNG Search APISemaDBSerpAPIShale ProtocolSingleStoreDBscikit-learnSlackSnowflakespaCySparkSparkLLMSpreedlySQLiteStack ExchangeStarRocksStochasticAIStreamlitStripeSupabase (Postgres)NebulaTairTelegramTencentTensorFlow DatasetsTiDBTigerGraphTigrisTogether AI2MarkdownTranswarpTrelloTrubricsTruLensTwitterTypesenseUnstructuredUpstageupstashUpTrainUSearchVDMSVearchVectaraVespavliteVoyageAIWandB TracingWeights & BiasesWeatherWeaviateWhatsAppWhyLabsWikipediaWolfram AlphaWriterxAIXataXorbits Inference (Xinference)YahooYandexYeager.aiYellowbrick01.AIYouYouTubeZepZhipu AIZillizComponentsChat modelsChat modelsAI21 LabsAlibaba Cloud PAI EASAnthropic[Deprecated] Experimental Anthropic Tools WrapperAnyscaleAzure OpenAIAzure ML EndpointBaichuan ChatBaidu QianfanAWS BedrockCerebrasCloudflare Workers AICohereCoze ChatDappier AIDatabricksDeepInfraEden AIErnie Bot ChatEverlyAIFireworksFriendliGigaChatGoogle AIGoogle Cloud Vertex AIGPTRouterGroqChatHuggingFaceIBM watsonx.aiJinaChatKineticaKonkoLiteLLMLiteLLM RouterLlama 2 ChatLlama APILlamaEdgeLlama.cppmaritalkMiniMaxMistralAIMLXMoonshotNaverNVIDIA AI EndpointsChatOCIModelDeploymentOCIGenAIChatOctoAIOllamaOpenAIOutlinesPerplexityPremAIPromptLayer ChatOpenAIRekaSambaNovaCloudSambaStudioSnowflake CortexsolarSparkLLM ChatNebula (Symbl.ai)Tencent HunyuanTogetherTongyi QwenUpstagevLLM ChatVolc Enging MaasWriterxAIYandexGPTChatYIYuan2.0ZHIPU AIRetrieversRetrieversActiveloop Deep MemoryAmazon KendraArceeArxivAskNewsAzure AI SearchBedrock (Knowledge Bases)BM25BoxBREEBS (Open Knowledge)ChaindeskChatGPT pluginCohere rerankerCohere RAGDocArrayDriaElasticSearch BM25ElasticsearchEmbedchainFlashRank rerankerFleet AI ContextGoogle DriveGoogle Vertex AI SearchIBM watsonx.aiJaguarDB Vector DatabaseKay.aiKinetica Vectorstore based RetrieverkNNLinkupSearchRetrieverLLMLingua Document CompressorLOTR (Merger Retriever)MetalMilvus Hybrid SearchNanoPQ (Product Quantization)needleOutlinePinecone Hybrid SearchPubMedQdrant Sparse VectorRAGatouilleRePhraseQueryRememberizerSEC filingSelf-querying retrieversSingleStoreDBSVMTavilySearchAPITF-IDF**NeuralDB**VespaWikipediaYou.comZep CloudZep Open SourceZilliz Cloud PipelineTools/ToolkitsToolsAINetwork ToolkitAlpha VantageAmadeus ToolkitArXivAskNewsAWS LambdaAzure AI Services ToolkitAzure Cognitive Services ToolkitAzure Container Apps dynamic sessionsShell (bash)Bearly Code InterpreterBing SearchBrave SearchCassandra Database ToolkitCDPChatGPT PluginsClickUp ToolkitCogniswitch ToolkitConnery Toolkit and ToolsDall-E Image GeneratorDatabricks Unity Catalog (UC)DataForSEODataheraldDuckDuckGo SearchE2B Data AnalysisEden AIEleven Labs Text2SpeechExa SearchFile SystemFinancialDatasets ToolkitGithub ToolkitGitlab ToolkitGmail ToolkitGolden QueryGoogle BooksGoogle Cloud Text-to-SpeechGoogle DriveGoogle FinanceGoogle ImagenGoogle JobsGoogle LensGoogle PlacesGoogle ScholarGoogle SearchGoogle SerperGoogle TrendsGradioGraphQLHuggingFace Hub ToolsHuman as a toolIFTTT WebHooksInfobipIonic Shopping ToolJina SearchJira ToolkitJSON ToolkitLemon AgentLinkupSearchToolMemorizeMojeek SearchMultiOn ToolkitNASA ToolkitNuclia UnderstandingNVIDIA Riva: ASR and TTSOffice365 ToolkitOpenAPI ToolkitNatural Language API ToolkitsOpenWeatherMapOracle AI Vector Search: Generate SummaryPandas DataframePassio NutritionAIPlayWright Browser ToolkitPolygon IO Toolkit and ToolsPowerBI ToolkitPubMedPython REPLReddit SearchRequests ToolkitRiza Code InterpreterRobocorp ToolkitSceneXplainScrapeGraphSearchApiSearxNG SearchSemantic Scholar API ToolSerpAPISlack ToolkitSpark SQL ToolkitSQLDatabase ToolkitStackExchangeSteam ToolkitTavily SearchTwilioUpstageWikidataWikipediaWolfram AlphaYahoo Finance NewsYou.com SearchYouTubeZapier Natural Language ActionsZenGuard AIDocument loadersDocument loadersacreomAirbyteLoaderAirbyte CDK (Deprecated)Airbyte Gong (Deprecated)Airbyte Hubspot (Deprecated)Airbyte JSON (Deprecated)Airbyte Salesforce (Deprecated)Airbyte Shopify (Deprecated)Airbyte Stripe (Deprecated)Airbyte Typeform (Deprecated)Airbyte Zendesk Support (Deprecated)AirtableAlibaba Cloud MaxComputeAmazon TextractApify DatasetArcGISArxivLoaderAssemblyAI Audio TranscriptsAstraDBAsync ChromiumAsyncHtmlAthenaAWS S3 DirectoryAWS S3 FileAZLyricsAzure AI DataAzure Blob Storage ContainerAzure Blob Storage FileAzure AI Document IntelligenceBibTeXBiliBiliBlackboardBlockchainBoxBrave SearchBrowserbaseBrowserlessBSHTMLLoaderCassandraChatGPT DataCollege ConfidentialConcurrent LoaderConfluenceCoNLL-UCopy PasteCouchbaseCSVCube Semantic LayerDatadog LogsDedocDiffbotDiscordDocugamiDocusaurusDropboxDuckDBEmailEPubEtherscanEverNoteexample_dataFacebook ChatFaunaFigmaFireCrawlGeopandasGitGitBookGitHubGlue CatalogGoogle AlloyDB for PostgreSQLGoogle BigQueryGoogle BigtableGoogle Cloud SQL for SQL serverGoogle Cloud SQL for MySQLGoogle Cloud SQL for PostgreSQLGoogle Cloud Storage DirectoryGoogle Cloud Storage FileGoogle Firestore in Datastore ModeGoogle DriveGoogle El Carro for Oracle WorkloadsGoogle Firestore (Native Mode)Google Memorystore for RedisGoogle SpannerGoogle Speech-to-Text Audio TranscriptsGrobidGutenbergHacker NewsHuawei OBS DirectoryHuawei OBS FileHuggingFace datasetiFixitImagesImage captionsIMSDbIuguJoplinJSONLoaderJupyter NotebookKineticalakeFSLangSmithLarkSuite (FeiShu)LLM SherpaMastodonMathPixPDFLoaderMediaWiki DumpMerge Documents LoadermhtmlMicrosoft ExcelMicrosoft OneDriveMicrosoft OneNoteMicrosoft PowerPointMicrosoft SharePointMicrosoft WordNear BlockchainModern TreasuryMongoDBNeedle Document LoaderNews URLNotion DB 2/2NucliaObsidianOpen Document Format (ODT)Open City DataOracle Autonomous DatabaseOracle AI Vector Search: Document ProcessingOrg-modePandas DataFrameparsersPDFMinerPDFPlumberPebblo Safe DocumentLoaderPolars DataFramePsychicPubMedPyMuPDFPyPDFDirectoryLoaderPyPDFium2LoaderPyPDFLoaderPySparkQuipReadTheDocs DocumentationRecursive URLRedditRoamRocksetrspaceRSS FeedsRSTscrapflyScrapingAntSitemapSlackSnowflakeSource CodeSpiderSpreedlyStripeSubtitleSurrealDBTelegramTencent COS DirectoryTencent COS FileTensorFlow DatasetsTiDB2MarkdownTOMLTrelloTSVTwitterUnstructuredUnstructuredMarkdownLoaderUnstructuredPDFLoaderUpstageURLVsdxWeatherWebBaseLoaderWhatsApp ChatWikipediaUnstructuredXMLLoaderXorbits Pandas DataFrameYouTube audioYouTube transcriptsYuqueZeroxPDFLoaderVector storesVector storesActiveloop Deep LakeAerospikeAlibaba Cloud OpenSearchAnalyticDBAnnoyApache DorisApertureDBAstra DB Vector StoreAtlasAwaDBAzure Cosmos DB Mongo vCoreAzure Cosmos DB No SQLAzure AI SearchBagelBagelDBBaidu Cloud ElasticSearch VectorSearchBaidu VectorDBApache CassandraChromaClarifaiClickHouseCouchbaseDashVectorDatabricksDingoDBDocArray HnswSearchDocArray InMemorySearchAmazon Document DBDuckDBChina Mobile ECloud ElasticSearch VectorSearchElasticsearchEpsillaFaissFaiss (Async)Google AlloyDB for PostgreSQLGoogle BigQuery Vector SearchGoogle Cloud SQL for MySQLGoogle Cloud SQL for PostgreSQLFirestoreGoogle Memorystore for RedisGoogle SpannerGoogle Vertex AI Feature StoreGoogle Vertex AI Vector SearchHippoHologresInfinispanJaguar Vector DatabaseKDB.AIKineticaLanceDBLanternLLMRailsManticoreSearch VectorStoreMarqoMeilisearchAmazon MemoryDBMilvusMomento Vector Index (MVI)MongoDB AtlasMyScaleNeo4j Vector IndexNucliaDBOpenSearchOracle AI Vector Search: Vector StorePathwayPostgres EmbeddingPGVecto.rsPGVectorPineconeQdrantRedisRelytRocksetSAP HANA Cloud Vector EngineScaNNSemaDBSingleStoreDBscikit-learnSQLiteVecSQLite-VSSSQLServerStarRocksSupabase (Postgres)SurrealDBTairTencent Cloud VectorDBThirdAI NeuralDBTiDB VectorTigrisTileDBTimescale Vector (Postgres)TypesenseUpstash VectorUSearchValdIntel\'s Visual Data Management System (VDMS)VearchVectaraVespaviking DBvliteWeaviateXataYellowbrickZepZep CloudZillizEmbedding modelsEmbedding modelsAI21Aleph AlphaAnyscaleascendAwaDBAzureOpenAIBaichuan Text EmbeddingsBaidu QianfanBedrockBGE on Hugging FaceBookend AIClarifaiCloudflare Workers AIClova EmbeddingsCohereDashScopeDatabricksDeepInfraEDEN AIElasticsearchEmbaasERNIEFake EmbeddingsFastEmbed by QdrantFireworksGigaChatGoogle Generative AI EmbeddingsGoogle Vertex AIGPT4AllGradientHugging FaceIBM watsonx.aiInfinityInstruct Embeddings on Hugging FaceIPEX-LLM: Local BGE Embeddings on Intel CPUIPEX-LLM: Local BGE Embeddings on Intel GPUIntel¬Æ Extension for Transformers Quantized Text EmbeddingsJinaJohn Snow LabsLASER Language-Agnostic SEntence Representations Embeddings by Meta AILlama.cppllamafileLLMRailsLocalAIMiniMaxMistralAImodel2vecModelScopeMosaicMLNaverNLP CloudNomicNVIDIA NIMsOracle Cloud Infrastructure Generative AIOllamaOpenClipOpenAIOpenVINOEmbedding Documents using Optimized and Quantized EmbeddersOracle AI Vector Search: Generate EmbeddingsOVHcloudPinecone EmbeddingsPremAISageMakerSambaNovaSelf HostedSentence Transformers on Hugging FaceSolarSpaCySparkLLM Text EmbeddingsTensorFlow HubText Embeddings InferenceTextEmbed - Embedding Inference ServerTitan TakeoffTogether AIUpstageVolc EngineVoyage AIXorbits inference (Xinference)YandexGPTZhipuAIOtherComponentsDocument loadersWebBaseLoaderOn this pageWebBaseLoader\nThis covers how to use WebBaseLoader to load all text from HTML webpages into a document format that we can use downstream. For more custom logic for loading webpages look at some child class examples such as IMSDbLoader, AZLyricsLoader, and CollegeConfidentialLoader.\nIf you don\'t want to worry about website crawling, bypassing JS-blocking sites, and data cleaning, consider using FireCrawlLoader or the faster option SpiderLoader.\nOverview‚Äã\nIntegration details‚Äã\n\nTODO: Fill in table features.\nTODO: Remove JS support link if not relevant, otherwise ensure link is correct.\nTODO: Make sure API reference links are correct.\n\nClassPackageLocalSerializableJS supportWebBaseLoaderlangchain_community‚úÖ‚ùå‚ùå\nLoader features‚Äã\nSourceDocument Lazy LoadingNative Async SupportWebBaseLoader‚úÖ‚úÖ\nSetup‚Äã\nCredentials‚Äã\nWebBaseLoader does not require any credentials.\nInstallation‚Äã\nTo use the WebBaseLoader you first need to install the langchain-community python package.\n%pip install -qU langchain_community beautifulsoup4\nInitialization‚Äã\nNow we can instantiate our model object and load documents:\nfrom langchain_community.document_loaders import WebBaseLoaderloader = WebBaseLoader("https://www.espn.com/")API Reference:WebBaseLoader\nTo bypass SSL verification errors during fetching, you can set the "verify" option:\nloader.requests_kwargs = {\'verify\':False}\nInitialization with multiple pages‚Äã\nYou can also pass in a list of pages to load from.\nloader_multiple_pages = WebBaseLoader(["https://www.espn.com/", "https://google.com"])\nLoad‚Äã\ndocs = loader.load()docs[0]\nDocument(metadata={\'source\': \'https://www.espn.com/\', \'title\': \'ESPN - Serving Sports Fans. Anytime. Anywhere.\', \'description\': \'Visit ESPN for live scores, highlights and sports news. Stream exclusive games on ESPN+ and play fantasy sports.\', \'language\': \'en\'}, page_content="\\n\\n\\n\\n\\n\\n\\n\\n\\nESPN - Serving Sports Fans. Anytime. Anywhere.\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n        Skip to main content\\n    \\n\\n        Skip to navigation\\n    \\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n<\\n\\n>\\n\\n\\n\\n\\n\\n\\n\\n\\n\\nMenuESPN\\n\\n\\n\\n\\n\\nscores\\n\\n\\n\\nNFLNBAMLBOlympicsSoccerWNBA‚Ä¶BoxingCFLNCAACricketF1GolfHorseLLWSMMANASCARNBA G LeagueNBA Summer LeagueNCAAFNCAAMNCAAWNHLNWSLPLLProfessional WrestlingRacingRN BBRN FBRugbySports BettingTennisX GamesUFLMore ESPNFantasyWatchESPN BETESPN+\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n  \\n\\nSubscribe Now\\n\\n\\n\\n\\n\\nBoxing: Crawford vs. Madrimov (ESPN+ PPV)\\n\\n\\n\\n\\n\\n\\n\\nPFL Playoffs: Heavyweights & Women\'s Flyweights\\n\\n\\n\\n\\n\\n\\n\\nMLB\\n\\n\\n\\n\\n\\n\\n\\nLittle League Baseball: Regionals\\n\\n\\n\\n\\n\\n\\n\\nIn The Arena: Serena Williams\\n\\n\\n\\n\\n\\n\\n\\nThe 30 College Football Playoff Contenders\\n\\n\\nQuick Links\\n\\n\\n\\n\\n2024 Paris Olympics\\n\\n\\n\\n\\n\\n\\n\\nOlympics: Everything to Know\\n\\n\\n\\n\\n\\n\\n\\nMLB Standings\\n\\n\\n\\n\\n\\n\\n\\nSign up: Fantasy Football\\n\\n\\n\\n\\n\\n\\n\\nWNBA Rookie Tracker\\n\\n\\n\\n\\n\\n\\n\\nNBA Free Agency Buzz\\n\\n\\n\\n\\n\\n\\n\\nLittle League Baseball, Softball\\n\\n\\n\\n\\n\\n\\n\\nESPN Radio: Listen Live\\n\\n\\n\\n\\n\\n\\n\\nWatch Golf on ESPN\\n\\n\\n\\n\\n\\n\\nFavorites\\n\\n\\n\\n\\n\\n\\n      Manage Favorites\\n      \\n\\n\\n\\nCustomize ESPNCreate AccountLog InFantasy\\n\\n\\n\\n\\nFootball\\n\\n\\n\\n\\n\\n\\n\\nBaseball\\n\\n\\n\\n\\n\\n\\n\\nBasketball\\n\\n\\n\\n\\n\\n\\n\\nHockey\\n\\n\\nESPN Sites\\n\\n\\n\\n\\nESPN Deportes\\n\\n\\n\\n\\n\\n\\n\\nAndscape\\n\\n\\n\\n\\n\\n\\n\\nespnW\\n\\n\\n\\n\\n\\n\\n\\nESPNFC\\n\\n\\n\\n\\n\\n\\n\\nX Games\\n\\n\\n\\n\\n\\n\\n\\nSEC Network\\n\\n\\nESPN Apps\\n\\n\\n\\n\\nESPN\\n\\n\\n\\n\\n\\n\\n\\nESPN Fantasy\\n\\n\\n\\n\\n\\n\\n\\nTournament Challenge\\n\\n\\nFollow ESPN\\n\\n\\n\\n\\nFacebook\\n\\n\\n\\n\\n\\n\\n\\nX/Twitter\\n\\n\\n\\n\\n\\n\\n\\nInstagram\\n\\n\\n\\n\\n\\n\\n\\nSnapchat\\n\\n\\n\\n\\n\\n\\n\\nTikTok\\n\\n\\n\\n\\n\\n\\n\\nYouTube\\n\\n\\nCollege football\'s most entertaining conference? Why the 16-team Big 12 is wiiiiiide open this seasonLong known as one of the sport\'s most unpredictable conferences, the new-look Big 12 promises another dose of chaos.11hBill ConnellyScott Winters/Icon SportswireUSC, Oregon and the quest to bulk up for the Big TenTo improve on D, the Trojans wanted to bulk up for a new league. They\'re not the only team trying to do that.10hAdam RittenbergThe 30 teams that can reach the CFPConnelly\'s best games of the 2024 seasonTOP HEADLINESTeam USA sets world record in 4x400 mixed relayGermany beats France, undefeated in men\'s hoopsU.S. men\'s soccer exits Games after Morocco routHungary to protest Khelif\'s Olympic participationKobe\'s Staples Center locker sells for record $2.9MDjokovic, Alcaraz to meet again, this time for goldKerr: Team USA lineups based on players\' rolesMarchand wins 200m IM; McEvoy takes 50 freeScouting Shedeur Sanders\' NFL futureLATEST FROM PARISBreakout stars, best moments and what comes next: Breaking down the Games halfway throughAt the midpoint of the Olympics, we look back at some of the best moments and forward at what\'s still to come in the second week.35mESPNMustafa Yalcin/Anadolu via Getty ImagesThe numbers behind USA\'s world record in 4x400 mixed relay4h0:46Medal trackerFull resultsFull coverage of the OlympicsPRESEASON HAS BEGUN!New kickoff rules on display to start Hall of Fame Game19h0:41McAfee on NFL\'s new kickoff: It looks like a practice drill6h1:11NFL\'s new kickoff rules debut to mixed reviewsTexans-Bears attracts more bets than MLBTOP RANK BOXINGSATURDAY ON ESPN+ PPVWhy Terence Crawford is playing the long game for a chance to face CaneloCrawford is approaching Saturday\'s marquee matchup against Israil Madrimov with his sights set on landing a bigger one next: against Canelo Alvarez.10hMike CoppingerMark Robinson/Matchroom BoxingBradley\'s take: Crawford\'s power vs. Israil Madrimov\'s disciplined styleTimothy Bradley Jr. breaks down the junior middleweight title fight.2dTimothy Bradley Jr.Buy Crawford vs. Madrimov on ESPN+ PPVChance to impress for Madrimov -- and UzbekistanHOW FRIDAY WENTMORE FROM THE PARIS OLYMPICSGrant Fisher makes U.S. track history, Marchand wins 4th gold and more Friday at the Paris GamesSha\'Carri Richardson made her long-awaited Olympic debut during the women\'s 100m preliminary round on Friday. Here\'s everything else you might have missed from Paris.27mESPNGetty ImagesU.S. men\'s loss to Morocco is a wake-up call before World CupOutclassed by Morocco, the U.S. men\'s Olympic team can take plenty of lessons with the World Cup on the horizon.4hSam BordenFull coverage of the OlympicsOLYMPIC MEN\'S HOOPS SCOREBOARDFRIDAY\'S GAMESOLYMPIC STANDOUTSWhy Simone Biles is now Stephen A.\'s No. 1 Olympian7h0:58Alcaraz on the cusp of history after securing spot in gold medal match8h0:59Simone Biles\' gymnastics titles: Olympics, Worlds, more statsOLYMPIC MEN\'S SOCCER SCOREBOARDFRIDAY\'S GAMESTRADE DEADLINE FALLOUTOlney: Eight big questions for traded MLB playersCan Jazz Chisholm Jr. handle New York? Is Jack Flaherty healthy? Will Jorge Soler\'s defense play? Key questions for players in new places.10hBuster OlneyWinslow Townson/Getty ImagesRanking the top prospects who changed teams at the MLB trade deadlineYou know the major leaguers who moved by now -- but what about the potential stars of tomorrow?1dKiley McDanielMLB Power RankingsSeries odds: Dodgers still on top; Phillies, Yanks behind them Top HeadlinesTeam USA sets world record in 4x400 mixed relayGermany beats France, undefeated in men\'s hoopsU.S. men\'s soccer exits Games after Morocco routHungary to protest Khelif\'s Olympic participationKobe\'s Staples Center locker sells for record $2.9MDjokovic, Alcaraz to meet again, this time for goldKerr: Team USA lineups based on players\' rolesMarchand wins 200m IM; McEvoy takes 50 freeScouting Shedeur Sanders\' NFL futureFavorites FantasyManage FavoritesFantasy HomeCustomize ESPNCreate AccountLog InICYMI0:47Nelson Palacio rips an incredible goal from outside the boxNelson Palacio scores an outside-of-the-box goal for Real Salt Lake in the 79th minute. \\n\\n\\nMedal Tracker\\n\\n\\n\\nCountries\\nAthletes\\n\\nOverall Medal Leaders43USA36FRA31CHNIndividual Medal LeadersGoldCHN 13FRA 11AUS 11SilverUSA 18FRA 12GBR 10BronzeUSA 16FRA 13CHN 9Overall Medal Leaders4MarchandMarchand3O\'CallaghanO\'Callaghan3McIntoshMcIntoshIndividual Medal LeadersGoldMarchand 4O\'Callaghan 3McIntosh 2SilverSmith 3Huske 2Walsh 2BronzeYufei 3Bhaker 2Haughey 2\\n\\n\\nFull Medal Tracker\\n\\n\\nBest of ESPN+ESPNCollege Football Playoff 2024: 30 teams can reach postseasonHeather Dinich analyzes the teams with at least a 10% chance to make the CFP.AP Photo/Ross D. FranklinNFL Hall of Fame predictions: Who will make the next 10 classes?When will Richard Sherman and Marshawn Lynch make it? Who could join Aaron Donald in 2029? Let\'s map out each Hall of Fame class until 2034.Thearon W. Henderson/Getty ImagesMLB trade deadline 2024: Ranking prospects who changed teamsYou know the major leaguers who moved by now -- but what about the potential stars of tomorrow who went the other way in those deals? Trending NowIllustration by ESPNRanking the top 100 professional athletes since 2000Who tops our list of the top athletes since 2000? We\'re unveiling the top 25, including our voters\' pick for the No. 1 spot.Photo by Kevin C. Cox/Getty Images2024 NFL offseason recap: Signings, coach moves, new rulesThink you missed something in the NFL offseason? We\'ve got you covered with everything important that has happened since February.Stacy Revere/Getty ImagesTop 25 college football stadiums: Rose Bowl, Michigan and moreFourteen of ESPN\'s college football writers rank the 25 best stadiums in the sport. Who\'s No. 1, who missed the cut and what makes these stadiums so special?ESPNInside Nate Robinson\'s silent battle -- and his fight to liveFor nearly 20 years Nate Robinson has been fighting a silent battle -- one he didn\'t realize until recently could end his life. Sign up to play the #1 Fantasy game!Create A LeagueJoin Public LeagueReactivate A LeagueMock Draft NowSign up for FREE!Create A LeagueJoin a Public LeagueReactivate a LeaguePractice With a Mock DraftSign up for FREE!Create A LeagueJoin a Public LeagueReactivate a LeaguePractice with a Mock DraftGet a custom ESPN experienceEnjoy the benefits of a personalized accountSelect your favorite leagues, teams and players and get the latest scores, news and updates that matter most to you. \\n\\nESPN+\\n\\n\\n\\n\\nBoxing: Crawford vs. Madrimov (ESPN+ PPV)\\n\\n\\n\\n\\n\\n\\n\\nPFL Playoffs: Heavyweights & Women\'s Flyweights\\n\\n\\n\\n\\n\\n\\n\\nMLB\\n\\n\\n\\n\\n\\n\\n\\nLittle League Baseball: Regionals\\n\\n\\n\\n\\n\\n\\n\\nIn The Arena: Serena Williams\\n\\n\\n\\n\\n\\n\\n\\nThe 30 College Football Playoff Contenders\\n\\n\\nQuick Links\\n\\n\\n\\n\\n2024 Paris Olympics\\n\\n\\n\\n\\n\\n\\n\\nOlympics: Everything to Know\\n\\n\\n\\n\\n\\n\\n\\nMLB Standings\\n\\n\\n\\n\\n\\n\\n\\nSign up: Fantasy Football\\n\\n\\n\\n\\n\\n\\n\\nWNBA Rookie Tracker\\n\\n\\n\\n\\n\\n\\n\\nNBA Free Agency Buzz\\n\\n\\n\\n\\n\\n\\n\\nLittle League Baseball, Softball\\n\\n\\n\\n\\n\\n\\n\\nESPN Radio: Listen Live\\n\\n\\n\\n\\n\\n\\n\\nWatch Golf on ESPN\\n\\n\\nFantasy\\n\\n\\n\\n\\nFootball\\n\\n\\n\\n\\n\\n\\n\\nBaseball\\n\\n\\n\\n\\n\\n\\n\\nBasketball\\n\\n\\n\\n\\n\\n\\n\\nHockey\\n\\n\\nESPN Sites\\n\\n\\n\\n\\nESPN Deportes\\n\\n\\n\\n\\n\\n\\n\\nAndscape\\n\\n\\n\\n\\n\\n\\n\\nespnW\\n\\n\\n\\n\\n\\n\\n\\nESPNFC\\n\\n\\n\\n\\n\\n\\n\\nX Games\\n\\n\\n\\n\\n\\n\\n\\nSEC Network\\n\\n\\nESPN Apps\\n\\n\\n\\n\\nESPN\\n\\n\\n\\n\\n\\n\\n\\nESPN Fantasy\\n\\n\\n\\n\\n\\n\\n\\nTournament Challenge\\n\\n\\nFollow ESPN\\n\\n\\n\\n\\nFacebook\\n\\n\\n\\n\\n\\n\\n\\nX/Twitter\\n\\n\\n\\n\\n\\n\\n\\nInstagram\\n\\n\\n\\n\\n\\n\\n\\nSnapchat\\n\\n\\n\\n\\n\\n\\n\\nTikTok\\n\\n\\n\\n\\n\\n\\n\\nYouTube\\n\\n\\nTerms of UsePrivacy PolicyYour US State Privacy RightsChildren\'s Online Privacy PolicyInterest-Based AdsAbout Nielsen MeasurementDo Not Sell or Share My Personal InformationContact UsDisney Ad Sales SiteWork for ESPNCorrectionsESPN BET is owned and operated by PENN Entertainment, Inc. and its subsidiaries (\'PENN\'). ESPN BET is available in states where PENN is licensed to offer sports wagering. Must be 21+ to wager. If you or someone you know has a gambling problem and wants help, call 1-800-GAMBLER.Copyright: ¬© 2024 ESPN Enterprises, Inc. All rights reserved.\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n")\nprint(docs[0].metadata)\n{\'source\': \'https://www.espn.com/\', \'title\': \'ESPN - Serving Sports Fans. Anytime. Anywhere.\', \'description\': \'Visit ESPN for live scores, highlights and sports news. Stream exclusive games on ESPN+ and play fantasy sports.\', \'language\': \'en\'}\nLoad multiple urls concurrently‚Äã\nYou can speed up the scraping process by scraping and parsing multiple urls concurrently.\nThere are reasonable limits to concurrent requests, defaulting to 2 per second.  If you aren\'t concerned about being a good citizen, or you control the server you are scraping and don\'t care about load, you can change the requests_per_second parameter to increase the max concurrent requests.  Note, while this will speed up the scraping process, but may cause the server to block you.  Be careful!\n%pip install -qU  nest_asyncio# fixes a bug with asyncio and jupyterimport nest_asyncionest_asyncio.apply()\nRequirement already satisfied: nest_asyncio in /Users/harrisonchase/.pyenv/versions/3.9.1/envs/langchain/lib/python3.9/site-packages (1.5.6)\nloader = WebBaseLoader(["https://www.espn.com/", "https://google.com"])loader.requests_per_second = 1docs = loader.aload()docs\n[Document(page_content="\\n\\n\\n\\n\\n\\n\\n\\n\\nESPN - Serving Sports Fans. Anytime. Anywhere.\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n        Skip to main content\\n    \\n\\n        Skip to navigation\\n    \\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n<\\n\\n>\\n\\n\\n\\n\\n\\n\\n\\n\\n\\nMenuESPN\\n\\n\\nSearch\\n\\n\\n\\nscores\\n\\n\\n\\nNFLNBANCAAMNCAAWNHLSoccer‚Ä¶MLBNCAAFGolfTennisSports BettingBoxingCFLNCAACricketF1HorseLLWSMMANASCARNBA G LeagueOlympic SportsRacingRN BBRN FBRugbyWNBAWorld Baseball ClassicWWEX GamesXFLMore ESPNFantasyListenWatchESPN+\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n  \\n\\nSUBSCRIBE NOW\\n\\n\\n\\n\\n\\nNHL: Select Games\\n\\n\\n\\n\\n\\n\\n\\nXFL\\n\\n\\n\\n\\n\\n\\n\\nMLB: Select Games\\n\\n\\n\\n\\n\\n\\n\\nNCAA Baseball\\n\\n\\n\\n\\n\\n\\n\\nNCAA Softball\\n\\n\\n\\n\\n\\n\\n\\nCricket: Select Matches\\n\\n\\n\\n\\n\\n\\n\\nMel Kiper\'s NFL Mock Draft 3.0\\n\\n\\nQuick Links\\n\\n\\n\\n\\nMen\'s Tournament Challenge\\n\\n\\n\\n\\n\\n\\n\\nWomen\'s Tournament Challenge\\n\\n\\n\\n\\n\\n\\n\\nNFL Draft Order\\n\\n\\n\\n\\n\\n\\n\\nHow To Watch NHL Games\\n\\n\\n\\n\\n\\n\\n\\nFantasy Baseball: Sign Up\\n\\n\\n\\n\\n\\n\\n\\nHow To Watch PGA TOUR\\n\\n\\n\\n\\n\\n\\nFavorites\\n\\n\\n\\n\\n\\n\\n      Manage Favorites\\n      \\n\\n\\n\\nCustomize ESPNSign UpLog InESPN Sites\\n\\n\\n\\n\\nESPN Deportes\\n\\n\\n\\n\\n\\n\\n\\nAndscape\\n\\n\\n\\n\\n\\n\\n\\nespnW\\n\\n\\n\\n\\n\\n\\n\\nESPNFC\\n\\n\\n\\n\\n\\n\\n\\nX Games\\n\\n\\n\\n\\n\\n\\n\\nSEC Network\\n\\n\\nESPN Apps\\n\\n\\n\\n\\nESPN\\n\\n\\n\\n\\n\\n\\n\\nESPN Fantasy\\n\\n\\nFollow ESPN\\n\\n\\n\\n\\nFacebook\\n\\n\\n\\n\\n\\n\\n\\nTwitter\\n\\n\\n\\n\\n\\n\\n\\nInstagram\\n\\n\\n\\n\\n\\n\\n\\nSnapchat\\n\\n\\n\\n\\n\\n\\n\\nYouTube\\n\\n\\n\\n\\n\\n\\n\\nThe ESPN Daily Podcast\\n\\n\\nAre you ready for Opening Day? Here\'s your guide to MLB\'s offseason chaosWait, Jacob deGrom is on the Rangers now? Xander Bogaerts and Trea Turner signed where? And what about Carlos Correa? Yeah, you\'re going to need to read up before Opening Day.12hESPNIllustration by ESPNEverything you missed in the MLB offseason3h2:33World Series odds, win totals, props for every teamPlay fantasy baseball for free!TOP HEADLINESQB Jackson has requested trade from RavensSources: Texas hiring Terry as full-time coachJets GM: No rush on Rodgers; Lamar not optionLove to leave North Carolina, enter transfer portalBelichick to angsty Pats fans: See last 25 yearsEmbiid out, Harden due back vs. Jokic, NuggetsLynch: Purdy \'earned the right\' to start for NinersMan Utd, Wrexham plan July friendly in San DiegoOn paper, Padres overtake DodgersLAMAR WANTS OUT OF BALTIMOREMarcus Spears identifies the two teams that need Lamar Jackson the most7h2:00Would Lamar sit out? Will Ravens draft a QB? Jackson trade request insightsLamar Jackson has asked Baltimore to trade him, but Ravens coach John Harbaugh hopes the QB will be back.3hJamison HensleyBallard, Colts will consider trading for QB JacksonJackson to Indy? Washington? Barnwell ranks the QB\'s trade fitsSNYDER\'S TUMULTUOUS 24-YEAR RUNHow Washington‚Äôs NFL franchise sank on and off the field under owner Dan SnyderSnyder purchased one of the NFL\'s marquee franchises in 1999. Twenty-four years later, and with the team up for sale, he leaves a legacy of on-field futility and off-field scandal.13hJohn KeimESPNIOWA STAR STEPS UP AGAINJ-Will: Caitlin Clark is the biggest brand in college sports right now8h0:47\'The better the opponent, the better she plays\': Clark draws comparisons to TaurasiCaitlin Clark\'s performance on Sunday had longtime observers going back decades to find comparisons.16hKevin PeltonWOMEN\'S ELITE EIGHT SCOREBOARDMONDAY\'S GAMESCheck your bracket!NBA DRAFTHow top prospects fared on the road to the Final FourThe 2023 NCAA tournament is down to four teams, and ESPN\'s Jonathan Givony recaps the players who saw their NBA draft stock change.11hJonathan GivonyAndy Lyons/Getty ImagesTALKING BASKETBALLWhy AD needs to be more assertive with LeBron on the court9h1:33Why Perk won\'t blame Kyrie for Mavs\' woes8h1:48WHERE EVERY TEAM STANDSNew NFL Power Rankings: Post-free-agency 1-32 poll, plus underrated offseason movesThe free agent frenzy has come and gone. Which teams have improved their 2023 outlook, and which teams have taken a hit?12hNFL Nation reportersIllustration by ESPNTHE BUCK STOPS WITH BELICHICKBruschi: Fair to criticize Bill Belichick for Patriots\' struggles10h1:27 Top HeadlinesQB Jackson has requested trade from RavensSources: Texas hiring Terry as full-time coachJets GM: No rush on Rodgers; Lamar not optionLove to leave North Carolina, enter transfer portalBelichick to angsty Pats fans: See last 25 yearsEmbiid out, Harden due back vs. Jokic, NuggetsLynch: Purdy \'earned the right\' to start for NinersMan Utd, Wrexham plan July friendly in San DiegoOn paper, Padres overtake DodgersFavorites FantasyManage FavoritesFantasy HomeCustomize ESPNSign UpLog InMarch Madness LiveESPNMarch Madness LiveWatch every men\'s NCAA tournament game live! ICYMI1:42Austin Peay\'s coach, pitcher and catcher all ejected after retaliation pitchAustin Peay\'s pitcher, catcher and coach were all ejected after a pitch was thrown at Liberty\'s Nathan Keeter, who earlier in the game hit a home run and celebrated while running down the third-base line. Men\'s Tournament ChallengeIllustration by ESPNMen\'s Tournament ChallengeCheck your bracket(s) in the 2023 Men\'s Tournament Challenge, which you can follow throughout the Big Dance. Women\'s Tournament ChallengeIllustration by ESPNWomen\'s Tournament ChallengeCheck your bracket(s) in the 2023 Women\'s Tournament Challenge, which you can follow throughout the Big Dance. Best of ESPN+AP Photo/Lynne SladkyFantasy Baseball ESPN+ Cheat Sheet: Sleepers, busts, rookies and closersYou\'ve read their names all preseason long, it\'d be a shame to forget them on draft day. The ESPN+ Cheat Sheet is one way to make sure that doesn\'t happen.Steph Chambers/Getty ImagesPassan\'s 2023 MLB season preview: Bold predictions and moreOpening Day is just over a week away -- and Jeff Passan has everything you need to know covered from every possible angle.Photo by Bob Kupbens/Icon Sportswire2023 NFL free agency: Best team fits for unsigned playersWhere could Ezekiel Elliott land? Let\'s match remaining free agents to teams and find fits for two trade candidates.Illustration by ESPN2023 NFL mock draft: Mel Kiper\'s first-round pick predictionsMel Kiper Jr. makes his predictions for Round 1 of the NFL draft, including projecting a trade in the top five. Trending NowAnne-Marie Sorvin-USA TODAY SBoston Bruins record tracker: Wins, points, milestonesThe B\'s are on pace for NHL records in wins and points, along with some individual superlatives as well. Follow along here with our updated tracker.Mandatory Credit: William Purnell-USA TODAY Sports2023 NFL full draft order: AFC, NFC team picks for all roundsStarting with the Carolina Panthers at No. 1 overall, here\'s the entire 2023 NFL draft broken down round by round. How to Watch on ESPN+Gregory Fisher/Icon Sportswire2023 NCAA men\'s hockey: Results, bracket, how to watchThe matchups in Tampa promise to be thrillers, featuring plenty of star power, high-octane offense and stellar defense.(AP Photo/Koji Sasahara, File)How to watch the PGA Tour, Masters, PGA Championship and FedEx Cup playoffs on ESPN, ESPN+Here\'s everything you need to know about how to watch the PGA Tour, Masters, PGA Championship and FedEx Cup playoffs on ESPN and ESPN+.Hailie Lynch/XFLHow to watch the XFL: 2023 schedule, teams, players, news, moreEvery XFL game will be streamed on ESPN+. Find out when and where else you can watch the eight teams compete. Sign up to play the #1 Fantasy Baseball GameReactivate A LeagueCreate A LeagueJoin a Public LeaguePractice With a Mock DraftSports BettingAP Photo/Mike KropfMarch Madness betting 2023: Bracket odds, lines, tips, moreThe 2023 NCAA tournament brackets have finally been released, and we have everything you need to know to make a bet on all of the March Madness games. Sign up to play the #1 Fantasy game!Create A LeagueJoin Public LeagueReactivateMock Draft Now\\n\\nESPN+\\n\\n\\n\\n\\nNHL: Select Games\\n\\n\\n\\n\\n\\n\\n\\nXFL\\n\\n\\n\\n\\n\\n\\n\\nMLB: Select Games\\n\\n\\n\\n\\n\\n\\n\\nNCAA Baseball\\n\\n\\n\\n\\n\\n\\n\\nNCAA Softball\\n\\n\\n\\n\\n\\n\\n\\nCricket: Select Matches\\n\\n\\n\\n\\n\\n\\n\\nMel Kiper\'s NFL Mock Draft 3.0\\n\\n\\nQuick Links\\n\\n\\n\\n\\nMen\'s Tournament Challenge\\n\\n\\n\\n\\n\\n\\n\\nWomen\'s Tournament Challenge\\n\\n\\n\\n\\n\\n\\n\\nNFL Draft Order\\n\\n\\n\\n\\n\\n\\n\\nHow To Watch NHL Games\\n\\n\\n\\n\\n\\n\\n\\nFantasy Baseball: Sign Up\\n\\n\\n\\n\\n\\n\\n\\nHow To Watch PGA TOUR\\n\\n\\nESPN Sites\\n\\n\\n\\n\\nESPN Deportes\\n\\n\\n\\n\\n\\n\\n\\nAndscape\\n\\n\\n\\n\\n\\n\\n\\nespnW\\n\\n\\n\\n\\n\\n\\n\\nESPNFC\\n\\n\\n\\n\\n\\n\\n\\nX Games\\n\\n\\n\\n\\n\\n\\n\\nSEC Network\\n\\n\\nESPN Apps\\n\\n\\n\\n\\nESPN\\n\\n\\n\\n\\n\\n\\n\\nESPN Fantasy\\n\\n\\nFollow ESPN\\n\\n\\n\\n\\nFacebook\\n\\n\\n\\n\\n\\n\\n\\nTwitter\\n\\n\\n\\n\\n\\n\\n\\nInstagram\\n\\n\\n\\n\\n\\n\\n\\nSnapchat\\n\\n\\n\\n\\n\\n\\n\\nYouTube\\n\\n\\n\\n\\n\\n\\n\\nThe ESPN Daily Podcast\\n\\n\\nTerms of UsePrivacy PolicyYour US State Privacy RightsChildren\'s Online Privacy PolicyInterest-Based AdsAbout Nielsen MeasurementDo Not Sell or Share My Personal InformationContact UsDisney Ad Sales SiteWork for ESPNCopyright: ¬© ESPN Enterprises, Inc. All rights reserved.\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n", lookup_str=\'\', metadata={\'source\': \'https://www.espn.com/\'}, lookup_index=0), Document(page_content=\'GoogleSearch Images Maps Play YouTube News Gmail Drive More ¬ªWeb History | Settings | Sign in\\xa0Advanced searchAdvertisingBusiness SolutionsAbout Google¬© 2023 - Privacy - Terms   \', lookup_str=\'\', metadata={\'source\': \'https://google.com\'}, lookup_index=0)]\nLoading a xml file, or using a different BeautifulSoup parser‚Äã\nYou can also look at SitemapLoader for an example of how to load a sitemap file, which is an example of using this feature.\nloader = WebBaseLoader(    "https://www.govinfo.gov/content/pkg/CFR-2018-title10-vol3/xml/CFR-2018-title10-vol3-sec431-86.xml")loader.default_parser = "xml"docs = loader.load()docs\n[Document(page_content=\'\\n\\n10\\nEnergy\\n3\\n2018-01-01\\n2018-01-01\\nfalse\\nUniform test method for the measurement of energy efficiency of commercial packaged boilers.\\n√Ç¬ß 431.86\\nSection √Ç¬ß 431.86\\n\\nEnergy\\nDEPARTMENT OF ENERGY\\nENERGY CONSERVATION\\nENERGY EFFICIENCY PROGRAM FOR CERTAIN COMMERCIAL AND INDUSTRIAL EQUIPMENT\\nCommercial Packaged Boilers\\nTest Procedures\\n\\n\\n\\n\\n¬ß\\u2009431.86\\nUniform test method for the measurement of energy efficiency of commercial packaged boilers.\\n(a) Scope. This section provides test procedures, pursuant to the Energy Policy and Conservation Act (EPCA), as amended, which must be followed for measuring the combustion efficiency and/or thermal efficiency of a gas- or oil-fired commercial packaged boiler.\\n(b) Testing and Calculations. Determine the thermal efficiency or combustion efficiency of commercial packaged boilers by conducting the appropriate test procedure(s) indicated in Table 1 of this section.\\n\\nTable 1‚ÄîTest Requirements for Commercial Packaged Boiler Equipment Classes\\n\\nEquipment category\\nSubcategory\\nCertified rated inputBtu/h\\n\\nStandards efficiency metric(¬ß\\u2009431.87)\\n\\nTest procedure(corresponding to\\nstandards efficiency\\nmetric required\\nby ¬ß\\u2009431.87)\\n\\n\\n\\nHot Water\\nGas-fired\\n‚â•300,000 and ‚â§2,500,000\\nThermal Efficiency\\nAppendix A, Section 2.\\n\\n\\nHot Water\\nGas-fired\\n>2,500,000\\nCombustion Efficiency\\nAppendix A, Section 3.\\n\\n\\nHot Water\\nOil-fired\\n‚â•300,000 and ‚â§2,500,000\\nThermal Efficiency\\nAppendix A, Section 2.\\n\\n\\nHot Water\\nOil-fired\\n>2,500,000\\nCombustion Efficiency\\nAppendix A, Section 3.\\n\\n\\nSteam\\nGas-fired (all*)\\n‚â•300,000 and ‚â§2,500,000\\nThermal Efficiency\\nAppendix A, Section 2.\\n\\n\\nSteam\\nGas-fired (all*)\\n>2,500,000 and ‚â§5,000,000\\nThermal Efficiency\\nAppendix A, Section 2.\\n\\n\\n\\u2003\\n\\n>5,000,000\\nThermal Efficiency\\nAppendix A, Section 2.OR\\nAppendix A, Section 3 with Section 2.4.3.2.\\n\\n\\n\\nSteam\\nOil-fired\\n‚â•300,000 and ‚â§2,500,000\\nThermal Efficiency\\nAppendix A, Section 2.\\n\\n\\nSteam\\nOil-fired\\n>2,500,000 and ‚â§5,000,000\\nThermal Efficiency\\nAppendix A, Section 2.\\n\\n\\n\\u2003\\n\\n>5,000,000\\nThermal Efficiency\\nAppendix A, Section 2.OR\\nAppendix A, Section 3. with Section 2.4.3.2.\\n\\n\\n\\n*\\u2009Equipment classes for commercial packaged boilers as of July 22, 2009 (74 FR 36355) distinguish between gas-fired natural draft and all other gas-fired (except natural draft).\\n\\n(c) Field Tests. The field test provisions of appendix A may be used only to test a unit of commercial packaged boiler with rated input greater than 5,000,000 Btu/h.\\n[81 FR 89305, Dec. 9, 2016]\\n\\n\\nEnergy Efficiency Standards\\n\\n\', lookup_str=\'\', metadata={\'source\': \'https://www.govinfo.gov/content/pkg/CFR-2018-title10-vol3/xml/CFR-2018-title10-vol3-sec431-86.xml\'}, lookup_index=0)]\nLazy Load‚Äã\nYou can use lazy loading to only load one page at a time in order to minimize memory requirements.\npages = []for doc in loader.lazy_load():    pages.append(doc)print(pages[0].page_content[:100])print(pages[0].metadata)\nESPN - Serving Sports Fans. Anytime. Anywhere.{\'source\': \'https://www.espn.com/\', \'title\': \'ESPN - Serving Sports Fans. Anytime. Anywhere.\', \'description\': \'Visit ESPN for live scores, highlights and sports news. Stream exclusive games on ESPN+ and play fantasy sports.\', \'language\': \'en\'}\nUsing proxies‚Äã\nSometimes you might need to use proxies to get around IP blocks. You can pass in a dictionary of proxies to the loader (and requests underneath) to use them.\nloader = WebBaseLoader(    "https://www.walmart.com/search?q=parrots",    proxies={        "http": "http://{username}:{password}:@proxy.service.com:6666/",        "https": "https://{username}:{password}:@proxy.service.com:6666/",    },)docs = loader.load()\nAPI reference‚Äã\nFor detailed documentation of all WebBaseLoader features and configurations head to the API reference: https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html\nRelated‚Äã\n\nDocument loader conceptual guide\nDocument loader how-to guides\nEdit this pageWas this page helpful?PreviousWeatherNextWhatsApp ChatOverviewIntegration detailsLoader featuresSetupCredentialsInstallationInitializationInitialization with multiple pagesLoadLoad multiple urls concurrentlyLoading a xml file, or using a different BeautifulSoup parserLazy LoadUsing proxiesAPI referenceRelatedCommunityTwitterGitHubOrganizationPythonJS/TSMoreHomepageBlogYouTubeCopyright ¬© 2024 LangChain, Inc.\n\n', metadata={'source': 'https://python.langchain.com/docs/integrations/document_loaders/web_base/', 'title': 'WebBaseLoader | \uf8ffü¶úÔ∏è\uf8ffüîó LangChain', 'description': 'This covers how to use WebBaseLoader to load all text from HTML webpages into a document format that we can use downstream. For more custom logic for loading webpages look at some child class examples such as IMSDbLoader, AZLyricsLoader, and CollegeConfidentialLoader.', 'language': 'en'})]



}
``` python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
```


#### ollama pull nomic-embed-text

###### Pull nomic-embed-text


}
``` python
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
```

 {.output .stream .stderr}
    c:\Users\Amr osama abdellatif\AppData\Local\Programs\Python\Python312\Lib\site-packages\onnxruntime\capi\onnxruntime_validation.py:26: UserWarning: Unsupported Windows version (11). ONNX Runtime supports Windows 10 and above, only.
      warnings.warn(



}
``` python
question="what can you tell me about web base loaders in langchain?"
docs = vectorstore.similarity_search(question)
len(docs)
```

 {.output .execute_result execution_count="14"}
    4



}
``` python
from langchain.chains import RetrievalQA
qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
res = qachain.invoke({"query": question})
print(res['result'])
```

 {.output .stream .stdout}
    Based on the provided context, here are some key points about WebBaseLoaders in LangChain:

    1. WebBaseLoader is a type of document loader in LangChain.
    2. It loads documents from websites by parsing HTML content using BeautifulSoup.
    3. You can load multiple pages concurrently by passing a list of URLs to the loader.
    4. The "verify" option allows bypassing SSL verification errors during fetching, but this should be used with caution as it may compromise security.
    5. WebBaseLoader does not require any credentials for authentication.
    6. To use WebBaseLoader, you need to install the langchain-community package and pip install BeautifulSoup4.

    If you have any specific questions about using WebBaseLoaders or want more information on their features, I'd be happy to try and help!


