import { TextLoader } from "langchain/document_loaders/fs/text";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import dotenv from 'dotenv';
import { RunnableSequence, RunnablePassthrough } from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import fs from 'fs';
import path from 'path';
import sgMail from '@sendgrid/mail';

dotenv.config();

export async function loadAndSplitChunks({ folderPath, chunkSize, chunkOverlap}) {
  const documents = [];
  const fileMap = new Map();

  const files = fs.readdirSync(folderPath);

  for (const file of files) {
    const filePath = path.join(folderPath, file);
    const extension = path.extname(filePath).toLowerCase();

    if (extension !== ".pdf" && extension !== ".txt") {
      console.log(`Skipping file: ${filePath} (Not a PDF or TXT)`);
      continue;
    }

    const loader = extension === ".pdf" ? new PDFLoader(filePath) : new TextLoader(filePath);
    const rawContent = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter({ chunkSize, chunkOverlap });
    const splitDoc = await splitter.splitDocuments(rawContent);
    documents.push(...splitDoc);
  }

  return  { documents, fileMap };
}

async function initializeVectorstoreWithDocuments({ documents }) {
  const embeddings = new OpenAIEmbeddings({ openAIApiKey: process.env.OPENAI_API_KEY });
  const vectorstore = new MemoryVectorStore(embeddings);
  await vectorstore.addDocuments(documents);
  return vectorstore;
}

function createDocumentRetrievalChain(retriever) {
  const convertDocsToString = (documents) => documents.map((doc) => `<doc>\n${doc.pageContent}\n</doc>`).join("\n");

  return RunnableSequence.from([
    (input) => input.standalone_question,
    retriever,
    convertDocsToString,
  ]);
}

function createRephraseQuestionChain() {
  const REPHRASE_QUESTION_SYSTEM_TEMPLATE = `meet the following objective to the best of your ability:`;

  const rephraseQuestionChainPrompt = ChatPromptTemplate.fromMessages([
    ["system", REPHRASE_QUESTION_SYSTEM_TEMPLATE],
    ["human", "Rephrase the following question or instruction to be standalone:\n{question}"],
  ]);

  return RunnableSequence.from([
    rephraseQuestionChainPrompt,
    new ChatOpenAI({ openAIApiKey: process.env.OPENAI_API_KEY, maxTokens: 2048 }),
    new StringOutputParser(),
  ]);
}

const ANSWER_CHAIN_SYSTEM_TEMPLATE = `You are a technical support assistant. Your goal is to provide personalized
 email responses to customers by retrieving relevant solutions from the provided sources
  and incorporating them into a friendly and helpful email response. Translate the email to {language}. use first person pronouns.
  Also, please sign off as AI support assistant. Your knowledge is limited to the information I  provide in the context. 
  You will answer this question based solely on this information, and you should not use any general knowledge or common sense.
  If the context provided cannot answer the user question, you will respond 'I don't have that information' in english without signing off.

  <context>
{context}
</context>

  The customer's issue description is: {standalone_question}`;


const answerGenerationChainPrompt = ChatPromptTemplate.fromMessages([
  ["system", ANSWER_CHAIN_SYSTEM_TEMPLATE],
  ["human", `Now, answer this question:\n{standalone_question}`],
]);

async function createConversationalRetrievalChain(retriever) {
  const rephraseQuestionChain = createRephraseQuestionChain();

  return RunnableSequence.from([
    RunnablePassthrough.assign({ standalone_question: rephraseQuestionChain }),
    RunnablePassthrough.assign({ context: createDocumentRetrievalChain(retriever) }),
    answerGenerationChainPrompt,
    new ChatOpenAI({ openAIApiKey: process.env.OPENAI_API_KEY, maxTokens: 2048 }),
  ]);
}

async function sendEmailResponse(email, response) {
  sgMail.setApiKey(process.env.SENDGRID_API_KEY);

  const responseString = response.content.replace(/\\n/g, '\n'); // Replace \\n with \n

  const msg = {
    to: email,
    from: 'admin@autoclaveglassworks.co.ke',
    subject: 'Response to your support request',
    text: responseString, // Use the string representation of the response
  };

  try {
    await sgMail.send(msg);
    console.log('Email sent successfully');
  } catch (error) {
    console.error('Error sending email:', error);
  }
}


export async function handleSupportRequest(email, issue, language) {
  const { documents } = await loadAndSplitChunks({
    folderPath: './docs',
    chunkSize: 1536,
    chunkOverlap: 128,
  });

  const selectedDocuments = documents;
  const selectedDocumentContent = selectedDocuments.map((doc) => doc.pageContent).join('\n');

  const vectorstore = await initializeVectorstoreWithDocuments({ documents: selectedDocuments });
  const retriever = vectorstore.asRetriever();

  console.log('Selected document content:', selectedDocumentContent);

  const finalRetrievalChain = await createConversationalRetrievalChain(retriever);
  const response = await finalRetrievalChain.invoke({ question: issue, context: selectedDocumentContent, language: language});
  console.log('language:', language);

  if (response.content.trim() === "I don't have that information.") {
    // Escalate to human-managed email
    await sendEmailToHumanSupport(email, issue);
  } else {
    await sendEmailResponse(email, response);
  }
}

async function sendEmailToHumanSupport(email, issue) {
  sgMail.setApiKey(process.env.SENDGRID_API_KEY);

  const msg = {
    to: 'sphinxspectre573@gmail.com', // Human support email
    from: 'admin@autoclaveglassworks.co.ke',
    subject: 'Escalated Support Request',
    text: `This issue has been escalated:\n\nCustomer Email: ${email}\nIssue: ${issue}`,
  };

  try {
    await sgMail.send(msg);
    console.log('Email sent to human support successfully');
  } catch (error) {
    console.error('Error sending email to human support:', error);
  }
}
