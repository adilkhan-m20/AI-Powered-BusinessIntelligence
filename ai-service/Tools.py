from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from io import BytesIO
import base64
import re
import matplotlib.pyplot as plt
import seaborn as sns


# Try to load FAISS database, create empty one if doesn't exist
try:
    db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    print("✅ Loaded existing FAISS database")
except Exception as e:
    print(f"⚠️ Could not load FAISS database: {e}")
    print("Creating empty FAISS database...")
    # Create a dummy document to initialize FAISS
    from langchain.schema import Document
    dummy_doc = Document(page_content="Initial document for FAISS initialization")
    db = FAISS.from_documents([dummy_doc], OpenAIEmbeddings())
    db.save_local("faiss_index")
    print("✅ Created empty FAISS database")
#db = FAISS.load_local("faiss_index",OpenAIEmbeddings(),allow_dangerous_deserialization=True)

# Now we create our retriever 
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # K is the amount of chunks to return
)
# retriever for graphs data
# mmr ensures the retrieved chunks cover more variety — useful if you want enough columns/rows for a meaningful chart.
retriever2 = db.as_retriever(
    search_type="mmr",  # Maximal Marginal Relevance — reduces redundancy
    search_kwargs={"k": 10, "lambda_mult": 0.5}
)

# function to convert the retrieved data from vector db into dataFrame
def get_data_from_vector_db(query: str) -> pd.DataFrame:
    """Retrieves data from Vector DB, parses it into a Pandas DataFrame.
    Supports CSV, TSV and Excel (.xls/.xlsx) formats."""

    docs = retriever2.invoke(query)
    if not docs:
        raise ValueError("No relevant data found in the database.")
    
    dataframes = []

    for doc in docs:
        content = doc.page_content.strip()

        # 1️ Try CSV/TSV parsing
        try:
            if "\t" in content:
                df = pd.read_csv(BytesIO(content.encode()), sep="\t")
            else:
                df = pd.read_csv(BytesIO(content.encode()))
            dataframes.append(df)
            continue
        except Exception:
            pass
        # 2 Try Excel parsing (base64 or file path)
        try:
            # Base64-encoded Excel
            if re.match(r"^[A-Za-z0-9+/=\s]+$", content) and len(content) > 50:
                excel_bytes = base64.b64decode(content)
                df = pd.read_excel(BytesIO(excel_bytes))
                dataframes.append(df)
                continue
            
            # Path to Excel (if your retriever stored paths in metadata)
            if content.lower().endswith((".xls", ".xlsx")):
                df = pd.read_excel(content)
                dataframes.append(df)
                continue
        except Exception:
            pass

        if not dataframes:
            raise ValueError("Could not parse retrieved documents into a DataFrame.")

    # Merge into single DataFrame
    final_df = pd.concat(dataframes, ignore_index=True)

    return final_df

# Utility to save and encode
def save_plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return f"![Graph](data:image/png;base64,{img_base64})"


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Vector DB.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the db."
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)

@tool
def histogram_tool(query: str) -> str:
    """This tools creates a histogram"""
    df = get_data_from_vector_db(query)
    num_col = df.select_dtypes(include="number").columns[0]
    plt.hist(df[num_col], bins=10, edgecolor="black")
    plt.title(f"Histogram of {num_col}")
    return save_plot_to_base64()

@tool
def scatter_tool(query: str) -> str:
    """This tool creates a Scatter plot"""
    df = get_data_from_vector_db(query)
    num_cols = df.select_dtypes(include="number").columns[:2]
    plt.scatter(df[num_cols[0]], df[num_cols[1]])
    plt.xlabel(num_cols[0])
    plt.ylabel(num_cols[1])
    plt.title(f"Scatter Plot: {num_cols[0]} vs {num_cols[1]}")
    return save_plot_to_base64()

@tool
def boxplot_tool(query: str) -> str:
    """This tool creates a BOX Plot"""
    df = get_data_from_vector_db(query)
    num_col = df.select_dtypes(include="number").columns[0]
    plt.boxplot(df[num_col].dropna())
    plt.title(f"Box Plot of {num_col}")
    return save_plot_to_base64()

@tool
def lineplot_tool(query: str) -> str:
    """This tool creates a Line Plot"""
    df = get_data_from_vector_db(query)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols].plot(kind="line")
    plt.title("Line Plot")
    return save_plot_to_base64()

@tool
def barchart_tool(query: str) -> str:
    """This tool creates a Bar Chart"""
    df = get_data_from_vector_db(query)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols].sum().plot(kind="bar")
    plt.title("Bar Chart")
    return save_plot_to_base64()

@tool
def piechart_tool(query: str) -> str:
    """This tool creates a PIE Chart"""
    df = get_data_from_vector_db(query)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols[0]].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title(f"Pie Chart of {num_cols[0]}")
    return save_plot_to_base64()

@tool
def seaborn_barplot_tool(query: str) -> str:
    """This tool creates a BAR Plot"""
    df = get_data_from_vector_db(query)
    num_cols = df.select_dtypes(include="number").columns
    sns.barplot(data=df, x=num_cols[0], y=num_cols[1])
    plt.title("Seaborn Barplot")
    return save_plot_to_base64()

@tool
def countplot_tool(query: str) -> str:
    """This tool creates a Count Plot"""
    df = get_data_from_vector_db(query)
    cat_col = df.select_dtypes(include="object").columns[0]
    sns.countplot(data=df, x=cat_col)
    plt.title("Count Plot")
    return save_plot_to_base64()

@tool
def violinplot_tool(query: str) -> str:
    """This tool creates a violin plot"""
    df = get_data_from_vector_db(query)
    num_cols = df.select_dtypes(include="number").columns
    sns.violinplot(data=df, x=num_cols[0], y=num_cols[1])
    plt.title("Violin Plot")
    return save_plot_to_base64()

@tool
def heatmap_tool(query: str) -> str:
    """This tool creates a Heat Map"""
    df = get_data_from_vector_db(query)
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    return save_plot_to_base64()

@tool
def pairplot_tool(query: str) -> str:
    """This tool creates a Pair Plot"""
    df = get_data_from_vector_db(query)
    pairplot_fig = sns.pairplot(df.select_dtypes(include="number"))
    buf = BytesIO()
    pairplot_fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return f"![Graph](data:image/png;base64,{img_base64})"

TOOLS = [pairplot_tool,heatmap_tool,violinplot_tool,countplot_tool,seaborn_barplot_tool,barchart_tool,piechart_tool,lineplot_tool,retriever_tool,histogram_tool,scatter_tool,boxplot_tool]

"""
If you print it, it will look something like:
{
    "messages": [
        SystemMessage(
            content="![Graph](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZAAAADsCAYAA...)"
        )
    ]
}
The content is Markdown with an embedded base64-encoded PNG.
which will be 
"""