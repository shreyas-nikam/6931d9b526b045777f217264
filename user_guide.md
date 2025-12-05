id: 6931d9b526b045777f217264_user_guide
summary: AI Design and Deployment Lab 2 - Clone User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# LLM Risk Assessment for Financial Summaries: A Codelab for Risk Managers

## Welcome to the LLM Risk Assessment Workbench
Duration: 00:05:00

<aside class="positive">
<b>Important Context:</b> As a Risk Manager at Apex Financial Services, your role is crucial in navigating the complexities of AI adoption. While Large Language Models (LLMs) offer immense potential for efficiency, especially in processing vast amounts of financial data, they also introduce inherent risks. This codelab will guide you through understanding, quantifying, and mitigating these emergent risks to ensure the trustworthiness and reliability of our LLM-powered financial summarization tool.
</aside>

Apex Financial Services aims to leverage LLMs to process financial news and reports, accelerating critical investment decisions. However, the inherent risks of LLMs – specifically **hallucinations** (factually incorrect but credible outputs) and **biases** (skewed interpretations or unfair representations) – pose significant threats. These emergent risks could lead to erroneous financial analyses, misinformed compliance strategies, and severe reputational damage. Your mission is to rigorously quantify these risks and formulate actionable mitigation strategies.

This interactive workbench is designed to empower you with the tools to systematically evaluate these risks. You will embark on a journey through critical steps:

1.  **Data Ingestion:** Load and review the source financial documents and ground truth.
2.  **Baseline Performance:** Understand the LLM's default behavior.
3.  **Hallucination Detection:** Quantify factual inaccuracies and contradictions.
4.  **Bias Identification:** Analyze sentiment and entity framing.
5.  **Adversarial Testing:** Proactively uncover vulnerabilities.
6.  **Prompt Refinement:** Improve LLM reliability.
7.  **Risk Aggregation & Recommendations:** Synthesize findings and advise on governance.

### Setting Up Your Workbench: Data Loading and Initialization

To begin our assessment, we need to load the financial data that the LLM will summarize and the corresponding human-verified ground truth data against which we will evaluate the LLM's performance. As a Risk Manager, ensuring the integrity and representativeness of this initial dataset is foundational to a robust risk assessment. This page allows you to either upload your own datasets or generate a set of dummy data to quickly proceed with the demonstration.

**How to Use:**
1.  **Option 1: Upload Your Data:** Use the "Upload Financial Documents (JSON)", "Upload Ground Truth Summaries (CSV)", and "Upload Ground Truth Facts (CSV)" file uploaders to provide your datasets.
2.  **Option 2: Generate Dummy Data:** For a quick start, click the "Generate Dummy Data (for demonstration)" button. This will create a small, pre-defined dataset for you.
3.  After either uploading or generating data, click the **"Load Data & Initialize"** button (if uploading) or the **"Generate Dummy Data (for demonstration)"** button (if generating).

<aside class="negative">
If you choose to upload your own data, ensure all three files (JSON for documents, CSV for summaries, CSV for facts) are provided. Missing files will result in an error.
</aside>

**Expected Output:**
Once data is successfully loaded, you will see a "Preview of Loaded Financial Documents" table. This table provides an excerpt of the document details, allowing you to quickly verify the data structure and content. A success message will confirm that the data is ready, and you can then proceed to the next step.

## Generating Baseline LLM Summaries
Duration: 00:03:00

### Understanding LLM Baseline Performance

Before diving into risk assessments, it's essential to establish a baseline of the LLM's performance. This step involves generating summaries from the LLM using a standard, unrefined prompt. As a Risk Manager, understanding this baseline helps you appreciate the model's initial capabilities and identify the areas where risks might naturally arise, serving as a benchmark for subsequent improvements.

**How to Use:**
1.  On this page, locate the **"Generate Baseline Summaries"** button.
2.  Click this button to instruct the application to process all loaded financial documents through a simulated LLM, generating a summary and a confidence score for each.

**Expected Output:**
After generation, you will see a confirmation message. You can then use the **"Select a document to review its baseline summary:"** dropdown to view the original document text, the ground truth summary, the LLM-generated summary, and its associated confidence score. This direct comparison is crucial for a Risk Manager to get a first impression of the LLM's default output quality.

## Hallucination Assessment: Factual Accuracy via Key Fact Extraction
Duration: 00:07:00

### Quantifying Factual Accuracy

A primary concern for financial summaries is their absolute factual accuracy. Hallucinations in reported financial figures can be catastrophic. As a Risk Manager, your task is to ensure that critical facts are correctly extracted and reported by the LLM. This section implements a systematic way to extract and compare these key facts from both the LLM's summary and the pre-defined ground truth.

We quantify factual accuracy using a ratio:
$$ Factual\_Accuracy = \frac{\text{Number of Correctly Extracted Facts}}{\text{Total Number of Ground Truth Facts}} $$
A fact is considered 'correctly extracted' if the LLM output contains the specified `fact_key` and its `fact_value` matches the ground truth (allowing for minor formatting variations). **Hallucinated facts** are those plausible financial figures in the LLM output that are not present in the ground truth.

### Configuring Fact Comparison

This step allows you to control how strictly the application matches facts extracted from the LLM summary against the ground truth. A higher threshold (e.g., 0.9) demands almost exact matches, while a lower one (e.g., 0.7) allows for more semantic flexibility. As a Risk Manager, this control helps you simulate different levels of strictness in reviewing financial reporting, directly impacting the calculated factual accuracy score.

**How to Use:**
1.  Adjust the **"Fact Comparison Threshold (Similarity Ratio)"** slider to set the strictness of fact matching. A value of 0.8 is a good starting point.
2.  Click the **"Assess Factual Accuracy"** button to run the assessment across all baseline summaries.

**Expected Output:**
The results will display an **"Overall Average Factual Accuracy"** metric. This score directly indicates the reliability of the LLM's financial reporting. A score below your acceptable threshold for financial reporting indicates a clear need for human intervention or further prompt engineering to mitigate hallucination risks.

Below this, a **"Detailed Fact Review per Document"** section allows you to select individual documents and see a table comparing ground truth facts with what the LLM found. It will also list the number of correct, missing, and potentially hallucinated facts, along with the document's individual factual accuracy score.

## Hallucination Assessment: Detecting Contradictions and Fabrications
Duration: 00:05:00

### Identifying Subtle Misrepresentations

Beyond simply missing or fabricating numbers, LLMs can generate text that subtly contradicts the source document or presents misleading information. As a Risk Manager, you need to identify these subtle yet dangerous misrepresentations. This involves a deeper semantic comparison between the LLM summary and the original document and ground truth summary.

### How this page helps you (Contradiction Detection)

This step allows you to trigger a specialized mechanism to detect statements in the LLM's summaries that contradict the original source document or the human-verified ground truth. Identifying these contradictions is crucial for uncovering more sophisticated forms of LLM hallucinations that might not be caught by simple fact-checking. This directly supports your role in ensuring the integrity and consistency of AI-generated financial insights.

**How to Use:**
1.  Click the **"Detect Contradictions"** button. The application will analyze all baseline LLM summaries against their original documents and ground truth summaries for inconsistencies.

**Expected Output:**
You will see a **"Total Detected Contradictions (across all documents)"** metric. Below that, the **"Detailed Contradiction Review per Document"** section allows you to select a document from a dropdown. For the selected document, you will see its original document excerpt, the LLM summary, and a list of any identified contradictions. These specific instances of contradiction require immediate attention, as they represent fundamental misrepresentations that could lead to erroneous interpretations or compliance issues.

## Quantifying Uncertainty: Aligning LLM Confidence with Accuracy
Duration: 00:06:00

### Assessing Trustworthiness of LLM Confidence

As a Risk Manager, you want to assess if the LLM's reported confidence is actually indicative of factual accuracy. A model that is confidently wrong is far more dangerous than one that knows its limitations. We will evaluate how well the LLM's self-reported confidence aligns with the empirically observed factual accuracy of its summaries.

We will calculate the Pearson correlation coefficient $r$ between the LLM's confidence scores ($X$) and the factual accuracy scores ($Y$) for each document:
$$ r = \frac{\sum_{i=1}^{N} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{N} (X_i - \bar{X})^2 \sum_{i=1}^{N} (Y_i - \bar{Y})^2}} $$

### How this page helps you (Confidence vs. Accuracy Evaluation)

This step directly helps you quantify the trustworthiness of the LLM's self-assessment. By clicking the button below, the application correlates the confidence scores assigned by the LLM to its summaries with their actual factual accuracy. As a Risk Manager, a high positive correlation indicates that the LLM generally 'knows what it knows', making its confidence scores a useful signal for identifying high-risk summaries. A low or negative correlation, however, would flag a critical risk, suggesting the LLM's confidence cannot be relied upon, necessitating broader human review regardless of reported certainty.

**How to Use:**
1.  Click the **"Evaluate Confidence-Accuracy Correlation"** button.

**Expected Output:**
You will see a scatter plot titled **"LLM Confidence vs. Factual Accuracy"**. Each point represents a document, showing how confident the LLM was in its summary versus how accurate that summary actually was. A red dashed line indicates a minimum acceptable accuracy threshold, helping to identify summaries that are both low confidence and low accuracy.

Below the plot, the **"Confidence-Accuracy Correlation (Pearson r)"** metric will be displayed. A low positive correlation here suggests the LLM's self-reported certainty is not a trustworthy indicator of its factual correctness, highlighting a significant oversight risk.

## Bias Detection: Analyzing Sentiment Skew Across Categories
Duration: 00:05:00

### Detecting Sentiment Distortions

Financial markets are highly sensitive to sentiment. Biases in LLM-generated summaries, such as consistently portraying certain industries more positively or negatively, can introduce distortions. As a Risk Manager, you need to detect if the LLM amplifies or introduces sentiment biases.

We define the sentiment discrepancy for a document $i$ as $SD_i = |S_{LLM,i} - S_{GT,i}|$, where $S_{LLM,i}$ is the LLM's sentiment score and $S_{GT,i}$ is the ground truth sentiment score.

### How this page helps you (Sentiment Bias Analysis)

This section allows you to categorize financial documents by attributes like 'industry' or 'company' and then compare the average sentiment of the LLM's summaries against the ground truth sentiment for each category. As a Risk Manager, this helps you identify systemic biases where the LLM consistently assigns a different sentiment to a particular sector or entity compared to human judgment. Detecting such a 'disparate impact' early is crucial for preventing skewed investment recommendations or misinformed compliance decisions.

**How to Use:**
1.  Select how you want to **"Categorize by:"** (e.g., "industry" or "company") using the dropdown.
2.  Click the **"Analyze Sentiment Bias"** button.

**Expected Output:**
A bar chart titled **"LLM vs. Ground Truth Sentiment by [Category]"** will be displayed, visually comparing the mean sentiment scores from the LLM summaries against the ground truth for each selected category. Below the chart, a detailed table provides the numerical breakdown. A significant divergence in sentiment between the LLM and ground truth for any category indicates a potential bias that warrants further investigation. Consistent positive or negative skew could lead to misinterpretations of financial health or market outlook within those categories.

## Bias Detection: Entity Representation and Framing Analysis
Duration: 00:06:00

### Ensuring Fair Entity Representation

Beyond overall sentiment, how an LLM references and frames specific entities can introduce subtle biases. Your role as a Risk Manager is to ensure the LLM presents information fairly and avoids perpetuating stereotypes. We will extract key entities and analyze their frequency and contextual sentiment in LLM summaries compared to ground truth.

### How this page helps you (Entity Framing Analysis)

This step allows you to specify key entities (e.g., specific companies, roles like 'CEO') and then run an analysis comparing how frequently the LLM mentions these entities and the average sentiment around them, versus their representation in the original documents and ground truth. As a Risk Manager, this helps you uncover subtle 'representational harms' – for instance, if the LLM consistently downplays certain entities or frames them negatively without factual basis. Such biases could lead to unfair valuations or misjudgment of market players.

**How to Use:**
1.  Use the **"Target Entities for Analysis (Optional)"** multiselect box to choose which entities you want to focus on. You can add or remove entities from the default list.
2.  Click the **"Analyze Entity Framing Bias"** button.

**Expected Output:**
A bar plot titled **"Top Entity Mention Counts: LLM vs. Ground Truth (Original Document + GT Summary)"** will illustrate the mention counts of selected entities in LLM summaries compared to the combined ground truth (original document + GT summary). Below the plot, a table provides a detailed breakdown including average contextual sentiment. A significant discrepancy in mention frequency or sentiment for key entities indicates a potential framing bias. For example, if the LLM mentions a competitor far less frequently or with a consistently negative sentiment than warranted by the source, this could influence strategic competitive analysis.

## Proactive Risk Identification: Adversarial Testing for Emergent Risks
Duration: 00:07:00

### Actively Uncovering Vulnerabilities

Proactive risk identification means actively trying to break the system. As a Risk Manager, you will now craft 'adversarial prompts' specifically designed to intentionally surface hidden biases or trigger hallucinations. This directly tests the LLM for 'emergent risks' like hallucinations and goal mis-specification – aspects that might not be caught by standard evaluation metrics.

### How this page helps you (Adversarial Testing)

This page empowers you to act as a 'red-teamer', simulating attempts to intentionally provoke undesirable behaviors from the LLM. By defining and executing adversarial prompts, you can uncover subtle vulnerabilities that might lead to biased or hallucinatory outputs under specific, challenging inputs. As a Risk Manager, understanding these failure modes is crucial for building robust safeguards, improving prompt guidelines, and ultimately enhancing the LLM's resilience to manipulation.

**How to Use:**
1.  Select a document for adversarial testing using the **"Select a document for adversarial testing:"** dropdown.
2.  Review or modify the three provided **"Adversarial Prompt"** text areas. These prompts are designed to challenge the LLM in different ways (e.g., induce negative focus, fabricate competitive advantages, or exaggerate precariousness).
3.  Click the **"Run Adversarial Tests"** button.

**Expected Output:**
For each adversarial prompt, the LLM's response will be displayed, along with a list of **"Identified Issues"**. Each identified issue highlights a specific vulnerability: a place where the LLM can be misled to produce biased or fabricated information. This informs your strategies for prompt engineering and model fine-tuning to enhance robustness against malicious or subtle input manipulations.

## Mitigating Risks: Prompt Engineering for Improved Reliability
Duration: 00:05:00

### Iterative Improvement through Prompt Refinement

This page provides the crucial opportunity to implement **prompt engineering** based on your insights from the preceding risk assessments. You will craft a more precise and restrictive prompt, and then re-evaluate the LLM's performance on a selected document. As a Risk Manager, this demonstrates the iterative process of mitigating AI risks: identifying vulnerabilities, designing corrective interventions (like improved prompts), and then quantifying the resulting improvements in factual accuracy and bias reduction. This hands-on refinement solidifies your understanding of how to guide LLM behavior for trustworthy outcomes.

**How to Use:**
1.  Select a document for prompt refinement retest using the **"Select a document for prompt refinement retest:"** dropdown. It is recommended to choose a document that previously showed significant issues.
2.  Modify the **"Refined Prompt Template"** text area. The default prompt is designed to be more strict and neutral. You can customize it further based on the specific risks you identified (e.g., adding instructions to avoid speculation, maintain neutrality, or only extract explicit facts).
3.  Click the **"Refine & Retest LLM"** button.

**Expected Output:**
You will see the **"New LLM Summary (Refined)"** generated with your updated prompt, along with its new confidence score. Crucially, a comparison table will show the **"Factual Accuracy Score"** and **"Sentiment Discrepancy"** for both the **"Baseline Value"** (from initial assessment) and the **"Refined Value"**. Observing improvements in these metrics directly validates the effectiveness of prompt engineering as a mitigation strategy against identified LLM risks.

## Synthesizing Insights: Aggregating Risks and Generating Governance Recommendations
Duration: 00:08:00

### Holistic View of LLM Risks and Actionable Policies

As a Risk Manager, you need a consolidated and clear view of all identified risks across the LLM summarization process. Now, it's time to synthesize these diverse findings into clear, actionable insights and visualizations. This aggregation will form the backbone of our 'LLM Risk Assessment Report'.

### How this page helps you (Risk Aggregation)

This section provides a holistic view of the LLM's risk profile by aggregating all the individual risk metrics you've assessed. By clicking **"Aggregate All Risks"**, you will see summary statistics and visualizations that paint a clear picture of the LLM's performance, highlighting areas of high risk (e.g., frequent hallucinations) and areas of strength. As a Risk Manager, this consolidated view is indispensable for reporting to stakeholders and prioritizing mitigation strategies effectively.

This section further empowers you to configure thresholds for what you deem acceptable levels of risk across key metrics. Based on these thresholds and the aggregated risk findings, the application will generate a set of concrete, actionable recommendations for HITL (Human-in-the-Loop) interventions and broader AI governance policies. This directly supports your responsibility to formulate and implement risk mitigation strategies that are tailored to Apex Financial Services' risk appetite and operational realities.

**How to Use:**
1.  Click the **"Aggregate All Risks"** button to compile all the findings from previous steps. This will generate summary metrics and visualizations.
2.  Review the key metrics (Overall Avg Factual Accuracy, Confidence-Accuracy Correlation, Sentiment Discrepancy, Total Hallucinated Facts, Total Missing Facts, Total Detected Contradictions, Total Adversarial Failures) and the accompanying charts.
3.  Adjust the **"Configure Recommendation Thresholds"** sliders and input box for "Min Acceptable Factual Accuracy (%)", "Max Acceptable Sentiment Discrepancy (Abs. Score)", and "Max Acceptable Adversarial Failures". These thresholds reflect Apex Financial Services' risk appetite.
4.  Click the **"Generate HITL & Governance Recommendations"** button.

**Expected Output:**
You will see several metrics summarizing the overall performance and risks. Two visualizations will be presented:
*   A bar chart titled **"Summary of Identified Hallucination & Bias Risk Incidents"** showing counts for various risk types.
*   A bar chart titled **"Mean Sentiment Scores by Industry: LLM vs. Ground Truth"** illustrating sentiment balance across different industries.

Finally, a list of **"Key Recommendations for Human-in-the-Loop & Governance"** will be generated based on your configured thresholds. These recommendations outline practical steps for integrating human oversight, audit mechanisms, and continuous monitoring into the LLM deployment lifecycle.

## Presenting Findings: Finalizing the LLM Risk Assessment Report
Duration: 00:03:00

### Compiling Your Comprehensive Risk Assessment

You have completed your comprehensive LLM risk assessment. The final, crucial step is to compile all these findings, analyses, and actionable recommendations into a coherent 'LLM Risk Assessment Report'. This report is the tangible deliverable for senior management and relevant stakeholders, articulating the LLM's risk posture and the strategies for responsible deployment.

### How this page helps you (Final Report Generation)

This final step compiles all your hard work into a comprehensive, stakeholder-ready report. By clicking the button below, the application synthesizes all the risk metrics, observations, and policy recommendations you've developed throughout the assessment. As a Risk Manager, this report is your definitive statement on the LLM's trustworthiness, enabling Apex Financial Services to make informed decisions about AI adoption, ensuring compliance, and maintaining investor confidence.

**How to Use:**
1.  Click the **"Generate Final Risk Assessment Report"** button.

**Expected Output:**
A full markdown-formatted **"LLM Risk Assessment Report"** will be displayed. This report includes an executive summary, key findings, detailed observations on various risk types (factual hallucinations, contradictions, sentiment bias, adversarial vulnerability), and the generated recommendations for mitigation and governance. This report empowers Apex Financial Services to move forward with AI adoption confidently, ensuring integrity, compliance, and ultimately, responsible innovation.
