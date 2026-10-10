from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()  # Load environment variables from .env file

def main():
    print("Hello from agenticai!")
    information = """
    Elon Reeve Musk (/ˈiːlɒn/ ⓘ EE-lon; born June 28, 1971) is a businessman,
    industrialist, and former public official who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. 
    Musk has been the wealthiest person in the world since 2025,
    and briefly became the only trillionaire (in terms of US dollars) in June 2026; as of October 2026, 
    Forbes estimates his net worth to be US$1 trillion.

    Born into the wealthy Musk family in Pretoria, South Africa, Musk emigrated in 1989 to Canada;
    he has Canadian citizenship since his mother was born there.
    He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California
    to pursue business ventures. 
    In 1995, Musk co-founded Zip2, a web software company. 
    Following its sale in 1999, he co-founded X.com, 
    an e-commerce payment system that merged with Confinity in March 2000 to form PayPal, 
    which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
    """

    information_place = """

    Darjeeling (/dɑːr.ˈdʒiː.lɪŋ/,[14] Nepali: [ˈdard͡ziliŋ], Bengali: [ˈdarˌdʒiliŋ]) is a city in the northernmost region of the Indian state of West Bengal. Located in the Eastern Himalayas, it has an average elevation of 2,045 metres (6,709 ft).[10] To the west of Darjeeling lies the easternmost province of Nepal, to the east the Kingdom of Bhutan, to the north the Indian state of Sikkim, and farther north the Tibet Autonomous Region of China. Bangladesh lies to the south and southeast, and most of the state of West Bengal lies to the south and southwest, connected to the Darjeeling region by a narrow tract. Kangchenjunga, the world's third-highest mountain, rises to the north and is prominently visible on clear days.[f][16]

In the early 19th century, during East India Company rule in India, Darjeeling was identified as a potential summer retreat for British officials, soldiers, and their families. The narrow mountain ridge was leased from the Kingdom of Sikkim, and eventually annexed to British India. Experimentation with growing tea on the slopes below Darjeeling was highly successful. Thousands of labourers were recruited chiefly from Nepal to clear the forests, build European-style cottages and work in the tea plantations. The widespread deforestation displaced the indigenous peoples. Residential schools were established in and around Darjeeling for the education of children of the domiciled British in India. By the late-19th century, a novel narrow-gauge mountain railway, the Darjeeling Himalayan Railway, was bringing summer residents into the town and carrying a freight of tea out for export to the world. After India's independence in 1947, as the British left Darjeeling, its cottages were purchased by wealthy Indians from the plains and its tea plantations by out-of-town Indian business owners and conglomerates.
"""

    noun = "place"

    summary_template = """
    given the information {information_place}, about a {noun}, i want you to create:
    1. a summary of the information in 3 sentences or less
    2. a list of 5 key points about the {noun} in bullet points
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information_place", "noun"],
        template=summary_template
    )

    llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information_place": information_place, "noun": noun})

    print(response.content)


if __name__ == "__main__":
    main()