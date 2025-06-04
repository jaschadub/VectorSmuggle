"""Decoy generation system for creating realistic cover traffic."""

import logging
import random
from typing import Any

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


class DecoyGenerator:
    """Generates realistic decoy embeddings and documents for cover traffic."""

    def __init__(
        self,
        decoy_ratio: float = 0.4,
        embedding_model: OpenAIEmbeddings | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        seed: int | None = None
    ):
        """
        Initialize decoy generator.

        Args:
            decoy_ratio: Ratio of decoy data to real data (0.0-1.0)
            embedding_model: OpenAI embeddings model for generating decoy embeddings
            chunk_size: Size of text chunks for document generation
            chunk_overlap: Overlap between chunks
            seed: Random seed for reproducible decoy generation
        """
        self.decoy_ratio = decoy_ratio
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.logger = logging.getLogger(__name__)

        # Predefined decoy content templates
        self.decoy_templates = self._load_decoy_templates()

    def _load_decoy_templates(self) -> dict[str, list[str]]:
        """Load predefined decoy content templates."""
        return {
            "business": [
                "Our quarterly revenue analysis shows consistent growth across all market segments.",
                "The strategic partnership with leading technology companies has enhanced our competitive position.",
                "Market research indicates strong demand for innovative solutions in the enterprise sector.",
                "Customer satisfaction metrics demonstrate the effectiveness of our service delivery model.",
                "The implementation of new operational processes has improved efficiency by 25%.",
                "Investment in research and development continues to drive product innovation.",
                "Our global expansion strategy focuses on emerging markets with high growth potential.",
                "The digital transformation initiative has modernized our core business processes.",
                "Sustainability practices have been integrated into all aspects of our operations.",
                "Employee engagement programs have resulted in improved retention rates."
            ],
            "technical": [
                "The microservices architecture provides scalability and maintainability benefits.",
                "Implementation of containerization has streamlined our deployment processes.",
                "Machine learning algorithms are being utilized to optimize system performance.",
                "Cloud infrastructure migration has reduced operational costs significantly.",
                "API gateway implementation ensures secure and efficient service communication.",
                "Database optimization techniques have improved query response times.",
                "Automated testing frameworks guarantee code quality and reliability.",
                "Monitoring and observability tools provide comprehensive system insights.",
                "Security protocols have been enhanced to meet industry compliance standards.",
                "DevOps practices enable continuous integration and deployment workflows."
            ],
            "research": [
                "Recent studies demonstrate the effectiveness of novel computational approaches.",
                "Experimental results validate the proposed theoretical framework.",
                "Data analysis reveals significant correlations between key variables.",
                "The methodology incorporates best practices from multiple research domains.",
                "Peer review feedback has been incorporated into the final research design.",
                "Statistical significance testing confirms the reliability of our findings.",
                "Literature review encompasses the most recent developments in the field.",
                "Collaborative research efforts have expanded the scope of our investigation.",
                "Reproducibility protocols ensure the validity of experimental procedures.",
                "Future research directions include exploration of emerging technologies."
            ],
            "general": [
                "The weather forecast indicates favorable conditions for outdoor activities.",
                "Local community events provide opportunities for social engagement.",
                "Educational programs offer skill development for professional advancement.",
                "Health and wellness initiatives promote a balanced lifestyle approach.",
                "Environmental conservation efforts contribute to sustainable development.",
                "Cultural diversity enriches our understanding of global perspectives.",
                "Technology adoption continues to transform traditional industries.",
                "Economic indicators suggest positive trends in market conditions.",
                "Social media platforms facilitate communication and information sharing.",
                "Innovation drives progress across multiple sectors of the economy."
            ]
        }

    def generate_decoy_text(
        self,
        category: str = "general",
        length: int = 500,
        num_documents: int = 1
    ) -> list[str]:
        """
        Generate decoy text documents.

        Args:
            category: Category of decoy content ('business', 'technical', 'research', 'general')
            length: Approximate length of each document in characters
            num_documents: Number of decoy documents to generate

        Returns:
            List of generated decoy text documents
        """
        if category not in self.decoy_templates:
            category = "general"

        templates = self.decoy_templates[category]
        documents = []

        for _ in range(num_documents):
            document_parts = []
            current_length = 0

            while current_length < length:
                # Select random template and add variations
                template = random.choice(templates)

                # Add some variation to the template
                variations = self._add_text_variations(template)
                document_parts.append(variations)
                current_length += len(variations)

                # Add connecting phrases
                if current_length < length * 0.8:
                    connector = random.choice([
                        " Furthermore, ",
                        " Additionally, ",
                        " Moreover, ",
                        " In addition, ",
                        " Subsequently, ",
                        " Consequently, "
                    ])
                    document_parts.append(connector)
                    current_length += len(connector)

            document = "".join(document_parts)
            documents.append(document[:length])  # Trim to exact length

        self.logger.debug(f"Generated {num_documents} decoy documents in category '{category}'")
        return documents

    def _add_text_variations(self, text: str) -> str:
        """Add variations to text to make it more realistic."""
        variations = [
            # Add specific numbers
            lambda t: t.replace("25%", f"{random.randint(15, 35)}%"),
            lambda t: t.replace("significant", random.choice(["substantial", "notable", "considerable"])),
            lambda t: t.replace("improved", random.choice(["enhanced", "optimized", "upgraded"])),
            lambda t: t.replace("effective", random.choice(["efficient", "successful", "productive"])),
            lambda t: t.replace("innovative", random.choice(["cutting-edge", "advanced", "modern"])),
        ]

        # Apply random variations
        for variation in random.sample(variations, min(2, len(variations))):
            text = variation(text)

        return text

    def generate_decoy_embeddings(
        self,
        num_embeddings: int,
        embedding_dim: int | None = None,
        similarity_target: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Generate decoy embeddings that appear realistic.

        Args:
            num_embeddings: Number of decoy embeddings to generate
            embedding_dim: Dimension of embeddings (inferred from similarity_target if provided)
            similarity_target: Optional target embeddings to make decoys similar to

        Returns:
            Array of decoy embeddings
        """
        if similarity_target is not None:
            embedding_dim = similarity_target.shape[1]

        if embedding_dim is None:
            embedding_dim = 1536  # Default OpenAI embedding dimension

        if similarity_target is not None:
            # Generate decoys similar to target embeddings
            target_mean = np.mean(similarity_target, axis=0)
            target_std = np.std(similarity_target, axis=0)

            # Add some noise to avoid exact matches
            noise_factor = 0.1
            decoy_embeddings = np.random.normal(
                target_mean,
                target_std + noise_factor,
                (num_embeddings, embedding_dim)
            )
        else:
            # Generate random embeddings with realistic distribution
            # Use a mixture of Gaussians to simulate realistic embedding space
            num_components = 5
            component_weights = np.random.dirichlet(np.ones(num_components))

            decoy_embeddings = np.zeros((num_embeddings, embedding_dim))

            for i in range(num_embeddings):
                # Select component
                np.random.choice(num_components, p=component_weights)

                # Generate embedding from selected component
                mean = np.random.normal(0, 0.1, embedding_dim)
                std = np.random.uniform(0.05, 0.2)

                embedding = np.random.normal(mean, std, embedding_dim)

                # Normalize to unit sphere (common for embeddings)
                embedding = embedding / np.linalg.norm(embedding)

                decoy_embeddings[i] = embedding

        self.logger.debug(f"Generated {num_embeddings} decoy embeddings")
        return decoy_embeddings

    def generate_decoy_embeddings_from_text(
        self,
        num_embeddings: int,
        category: str = "general"
    ) -> tuple[np.ndarray, list[str]]:
        """
        Generate decoy embeddings from generated text.

        Args:
            num_embeddings: Number of decoy embeddings to generate
            category: Category of decoy content

        Returns:
            Tuple of (embeddings_array, source_texts)
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model required for text-based decoy generation")

        # Generate decoy texts
        texts_per_embedding = max(1, num_embeddings // 10)  # Reuse some texts
        decoy_texts = self.generate_decoy_text(
            category=category,
            length=self.chunk_size,
            num_documents=texts_per_embedding
        )

        # Create text splitter for chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Generate chunks and embeddings
        all_chunks = []
        for text in decoy_texts:
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)

        # Select required number of chunks
        if len(all_chunks) < num_embeddings:
            # Repeat chunks if not enough
            multiplier = (num_embeddings // len(all_chunks)) + 1
            all_chunks = all_chunks * multiplier

        selected_chunks = random.sample(all_chunks, num_embeddings)

        # Generate embeddings with retry logic
        embeddings = []
        failed_chunks = 0

        for chunk in selected_chunks:
            max_retries = 3
            retry_count = 0
            embedding_success = False

            while retry_count < max_retries and not embedding_success:
                try:
                    # Test API connectivity first
                    if not hasattr(self.embedding_model, '_test_connection_done'):
                        test_embedding = self.embedding_model.embed_query("test")
                        if not test_embedding or len(test_embedding) == 0:
                            raise ValueError("API connectivity test failed")
                        self.embedding_model._test_connection_done = True

                    embedding = self.embedding_model.embed_query(chunk)

                    if not embedding or len(embedding) == 0:
                        raise ValueError("Empty embedding returned")

                    embeddings.append(embedding)
                    embedding_success = True

                except Exception as e:
                    retry_count += 1
                    self.logger.warning(f"Attempt {retry_count} failed for chunk embedding: {e}")

                    if retry_count >= max_retries:
                        self.logger.error(f"Failed to generate embedding after {max_retries} attempts, using fallback")
                        failed_chunks += 1
                        # Generate random embedding as fallback
                        random_embedding = np.random.normal(0, 0.1, 1536)
                        random_embedding = random_embedding / np.linalg.norm(random_embedding)
                        embeddings.append(random_embedding.tolist())
                        embedding_success = True
                    else:
                        # Wait before retry (exponential backoff)
                        import time
                        time.sleep(2 ** retry_count)

        if failed_chunks > 0:
            self.logger.warning(f"Used fallback embeddings for {failed_chunks}/{len(selected_chunks)} chunks")

        embeddings_array = np.array(embeddings)

        self.logger.info(f"Generated {len(embeddings)} decoy embeddings from text")
        return embeddings_array, selected_chunks

    def mix_with_decoys(
        self,
        real_embeddings: np.ndarray,
        real_texts: list[str] | None = None,
        decoy_category: str = "general"
    ) -> dict[str, Any]:
        """
        Mix real embeddings with decoy embeddings.

        Args:
            real_embeddings: Real embeddings to hide
            real_texts: Optional real texts corresponding to embeddings
            decoy_category: Category for decoy generation

        Returns:
            Dictionary containing mixed embeddings and metadata
        """
        num_real = real_embeddings.shape[0]
        num_decoys = int(num_real * self.decoy_ratio)

        if num_decoys == 0:
            return {
                "embeddings": real_embeddings,
                "texts": real_texts,
                "real_indices": list(range(num_real)),
                "decoy_indices": [],
                "metadata": {
                    "num_real": num_real,
                    "num_decoys": 0,
                    "decoy_ratio": 0.0
                }
            }

        # Generate decoy embeddings
        if self.embedding_model is not None:
            decoy_embeddings, decoy_texts = self.generate_decoy_embeddings_from_text(
                num_decoys, decoy_category
            )
        else:
            decoy_embeddings = self.generate_decoy_embeddings(
                num_decoys,
                embedding_dim=real_embeddings.shape[1],
                similarity_target=real_embeddings
            )
            decoy_texts = self.generate_decoy_text(
                category=decoy_category,
                length=self.chunk_size,
                num_documents=num_decoys
            )

        # Combine real and decoy embeddings
        all_embeddings = np.vstack([real_embeddings, decoy_embeddings])
        all_texts = (real_texts or []) + decoy_texts

        # Create random permutation
        indices = np.random.permutation(len(all_embeddings))
        mixed_embeddings = all_embeddings[indices]
        mixed_texts = [all_texts[i] for i in indices]

        # Track real and decoy indices
        real_indices = [i for i, idx in enumerate(indices) if idx < num_real]
        decoy_indices = [i for i, idx in enumerate(indices) if idx >= num_real]

        result = {
            "embeddings": mixed_embeddings,
            "texts": mixed_texts,
            "real_indices": real_indices,
            "decoy_indices": decoy_indices,
            "metadata": {
                "num_real": num_real,
                "num_decoys": num_decoys,
                "decoy_ratio": num_decoys / (num_real + num_decoys),
                "decoy_category": decoy_category,
                "total_embeddings": len(mixed_embeddings)
            }
        }

        self.logger.info(f"Mixed {num_real} real embeddings with {num_decoys} decoys")
        return result

    def extract_real_data(self, mixed_data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract real data from mixed embeddings.

        Args:
            mixed_data: Dictionary containing mixed embeddings and metadata

        Returns:
            Dictionary containing only real embeddings and texts
        """
        real_indices = mixed_data["real_indices"]
        mixed_embeddings = mixed_data["embeddings"]
        mixed_texts = mixed_data["texts"]

        real_embeddings = mixed_embeddings[real_indices]
        real_texts = [mixed_texts[i] for i in real_indices]

        result = {
            "embeddings": real_embeddings,
            "texts": real_texts,
            "metadata": {
                "extracted_count": len(real_indices),
                "original_total": mixed_data["metadata"]["total_embeddings"]
            }
        }

        self.logger.info(f"Extracted {len(real_indices)} real embeddings from mixed data")
        return result

    def get_decoy_statistics(self) -> dict[str, Any]:
        """Get statistics about decoy generation capabilities."""
        stats = {
            "decoy_ratio": self.decoy_ratio,
            "available_categories": list(self.decoy_templates.keys()),
            "templates_per_category": {
                category: len(templates)
                for category, templates in self.decoy_templates.items()
            },
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model_available": self.embedding_model is not None
        }

        return stats
