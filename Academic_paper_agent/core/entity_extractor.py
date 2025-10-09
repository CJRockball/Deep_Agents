#%%

# entity_extractor.py - Step 7: Entity Extraction and Relationship Mapping (FIXED)

import re
import hashlib
from typing import List, Dict, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import sqlite3

from database_setup import DatabaseManager

@dataclass
class Entity:
    """Represents an extracted entity"""
    id: str
    paper_id: str
    entity_text: str
    entity_type: str  # 'author', 'institution', 'concept', 'method', 'dataset', 'tool', 'metric'
    normalized_form: str
    confidence_score: float
    section_id: Optional[str] = None
    chunk_id: Optional[str] = None
    first_mention_position: int = 0
    mention_count: int = 1
    context_sentences: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EntityRelationship:
    """Represents a relationship between two entities"""
    id: str
    paper_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # 'uses', 'implements', 'compares', 'extends', 'evaluates', 'cites'
    confidence_score: float
    context: str
    section_id: Optional[str] = None
    relationship_strength: float = 1.0

@dataclass
class EntityExtractionResult:
    """Result of entity extraction process"""
    paper_id: str
    total_entities: int
    entities_by_type: Dict[str, int]
    total_relationships: int
    relationships_by_type: Dict[str, int]
    processing_errors: List[str]
    quality_score: float

class EntityExtractor:
    """
    Extracts entities and relationships from academic papers.
    This is Step 7 in the academic paper processing pipeline.
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager or DatabaseManager()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Entity patterns for different types
        self.entity_patterns = {
            'method': {
                'patterns': [
                    r'\b(neural network|deep learning|machine learning|CNN|RNN|LSTM|GRU|transformer|BERT|GPT)\b',
                    r'\b(algorithm|approach|method|technique|framework|model|architecture)\b',
                    r'\b(gradient descent|backpropagation|attention mechanism|self-attention)\b',
                    r'\b(classification|regression|clustering|reinforcement learning)\b'
                ],
                'context_keywords': ['propose', 'introduce', 'implement', 'develop', 'design']
            },
            'dataset': {
                'patterns': [
                    r'\b(ImageNet|CIFAR|MNIST|COCO|WikiText|Common Crawl|OpenWebText)\b',
                    r'\bdataset\b',
                    r'\bcorpus\b',
                    r'\bbenchmark\b'
                ],
                'context_keywords': ['train', 'evaluate', 'test', 'dataset', 'data']
            },
            'metric': {
                'patterns': [
                    r'\b(accuracy|precision|recall|F1|BLEU|ROUGE|perplexity|AUC|MAP)\b',
                    r'\b(loss|error|score|performance|evaluation)\b'
                ],
                'context_keywords': ['achieve', 'measure', 'evaluate', 'score', 'performance']
            },
            'tool': {
                'patterns': [
                    r'\b(PyTorch|TensorFlow|Keras|scikit-learn|Hugging Face|OpenAI|LangChain)\b',
                    r'\b(Python|R|MATLAB|Julia|framework|library|toolkit)\b'
                ],
                'context_keywords': ['use', 'implement', 'build', 'develop']
            },
            'concept': {
                'patterns': [
                    r'\b(artificial intelligence|AI|natural language processing|NLP|computer vision|CV)\b',
                    r'\b(supervised learning|unsupervised learning|semi-supervised|few-shot|zero-shot)\b',
                    r'\b(optimization|regularization|generalization|overfitting|underfitting)\b'
                ],
                'context_keywords': ['concept', 'idea', 'theory', 'principle']
            },
            'institution': {
                'patterns': [
                    r'\b(University|Institute|Laboratory|Lab|Research|Center|Centre)\b[^.]{0,50}',
                    r'\b(MIT|Stanford|Harvard|Berkeley|CMU|Google|Microsoft|Facebook|OpenAI)\b'
                ],
                'context_keywords': ['affiliation', 'university', 'research']
            }
        }

        # Relationship patterns
        self.relationship_patterns = {
            'uses': [
                r'(\w+)\s+(?:uses|utilizes|employs|applies)\s+(\w+)',
                r'using\s+(\w+)',
                r'based on\s+(\w+)'
            ],
            'implements': [
                r'implement(?:s|ed)?\s+(\w+)',
                r'(\w+)\s+implementation',
                r'develop(?:s|ed)?\s+(\w+)'
            ],
            'compares': [
                r'compar(?:e|es|ed)\s+(?:with|to|against)\s+(\w+)',
                r'(\w+)\s+(?:vs|versus)\s+(\w+)',
                r'baseline\s+(\w+)'
            ],
            'evaluates': [
                r'evaluat(?:e|es|ed)\s+(?:on|using)\s+(\w+)',
                r'test(?:s|ed)?\s+on\s+(\w+)',
                r'experiment(?:s)?\s+(?:with|on)\s+(\w+)'
            ],
            'extends': [
                r'extend(?:s|ed)?\s+(\w+)',
                r'build(?:s)?\s+(?:on|upon)\s+(\w+)',
                r'improv(?:e|es|ed)\s+(\w+)'
            ]
        }

    def extract_entities_and_relationships(self, paper_id: str) -> EntityExtractionResult:
        """Main entry point for entity extraction"""
        self.logger.info(f"Extracting entities and relationships for paper {paper_id}")

        try:
            # Create tables
            self._create_entity_tables()

            # Load content chunks from previous step
            chunks = self._load_content_chunks(paper_id)

            if not chunks:
                return self._create_empty_result(paper_id, "No content chunks found")

            self.logger.info(f"Processing {len(chunks)} content chunks")

            # Extract entities from chunks
            entities = self._extract_entities_from_chunks(paper_id, chunks)

            # Extract additional entities from citations (authors, institutions)
            citation_entities = self._extract_entities_from_citations(paper_id)
            entities.extend(citation_entities)

            # Deduplicate and normalize entities
            entities = self._deduplicate_entities(entities)

            self.logger.info(f"Found {len(entities)} unique entities")

            # Extract relationships between entities
            relationships = self._extract_relationships(paper_id, entities, chunks)

            self.logger.info(f"Found {len(relationships)} relationships")

            # Calculate statistics
            result = self._calculate_extraction_stats(paper_id, entities, relationships)

            # Save to database
            if entities:
                self._save_entities_to_database(entities)
            if relationships:
                self._save_relationships_to_database(relationships)

            # Update processing status
            self._update_processing_status(paper_id, 'entity_extraction', len(entities) > 0)

            self.logger.info(f"Entity extraction completed for {paper_id}: "
                           f"{result.total_entities} entities, "
                           f"{result.total_relationships} relationships extracted")

            return result

        except Exception as e:
            self.logger.error(f"Error in entity extraction for {paper_id}: {e}")
            import traceback
            traceback.print_exc()  # Print full error trace for debugging
            return self._create_empty_result(paper_id, f"Processing failed: {str(e)}")

    def _create_entity_tables(self):
        """Create entity and relationship tables"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            # Entities table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    paper_id VARCHAR(50),
                    entity_text TEXT NOT NULL,
                    entity_type VARCHAR(20),
                    normalized_form TEXT,
                    confidence_score REAL,
                    section_id TEXT,
                    chunk_id TEXT,
                    first_mention_position INTEGER,
                    mention_count INTEGER,
                    context_sentences TEXT,
                    aliases TEXT,
                    metadata_json TEXT,
                    created_at TEXT,
                    FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id)
                )
            """)

            # Entity relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entity_relationships (
                    id TEXT PRIMARY KEY,
                    paper_id VARCHAR(50),
                    source_entity_id TEXT,
                    target_entity_id TEXT,
                    relationship_type VARCHAR(20),
                    confidence_score REAL,
                    context TEXT,
                    section_id TEXT,
                    relationship_strength REAL,
                    created_at TEXT,
                    FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id),
                    FOREIGN KEY (source_entity_id) REFERENCES entities(id),
                    FOREIGN KEY (target_entity_id) REFERENCES entities(id)
                )
            """)

            conn.commit()

        except Exception as e:
            self.logger.error(f"Error creating entity tables: {e}")

    def _load_content_chunks(self, paper_id: str) -> List[Dict]:
        """Load content chunks from previous processing step"""

        conn = self.db_manager.get_sqlite_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, chunk_text, section_id, section_title, canonical_section, chunk_order
                FROM content_chunks 
                WHERE paper_id = ? 
                ORDER BY chunk_order
            """, (paper_id,))

            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    'id': row[0],
                    'text': row[1],
                    'section_id': row[2],
                    'section_title': row[3],
                    'canonical_section': row[4],
                    'order': row[5]
                })

            return chunks

        except Exception as e:
            self.logger.warning(f"Error loading chunks: {e}")
            return []

    def _extract_entities_from_chunks(self, paper_id: str, chunks: List[Dict]) -> List[Entity]:
        """Extract entities from content chunks"""

        entities = []

        for chunk_idx, chunk in enumerate(chunks):
            chunk_text = chunk['text']

            # Extract entities by type
            for entity_type, type_config in self.entity_patterns.items():
                for pattern in type_config['patterns']:
                    try:
                        matches = re.finditer(pattern, chunk_text, re.IGNORECASE)

                        for match in matches:
                            entity_text = match.group(0).strip()
                            normalized_form = entity_text.lower()

                            # Skip very short or common entities
                            if len(entity_text) < 3 or entity_text.lower() in ['the', 'and', 'or', 'in', 'on', 'at']:
                                continue

                            # Calculate confidence based on context
                            confidence = self._calculate_entity_confidence(
                                entity_text, chunk_text, type_config['context_keywords']
                            )

                            # Create entity
                            entity_id = self._generate_entity_id(paper_id, entity_type, normalized_form)

                            entity = Entity(
                                id=entity_id,
                                paper_id=paper_id,
                                entity_text=entity_text,
                                entity_type=entity_type,
                                normalized_form=normalized_form,
                                confidence_score=confidence,
                                section_id=chunk['section_id'],
                                chunk_id=chunk['id'],
                                first_mention_position=match.start(),
                                mention_count=1,
                                context_sentences=[self._extract_sentence_context(chunk_text, match.start())],
                                aliases=[],
                                metadata={
                                    'section_title': chunk['section_title'],
                                    'canonical_section': chunk['canonical_section']
                                }
                            )

                            entities.append(entity)

                    except Exception as e:
                        self.logger.warning(f"Error processing pattern {pattern} in chunk {chunk_idx}: {e}")
                        continue

        return entities

    def _extract_entities_from_citations(self, paper_id: str) -> List[Entity]:
        """Extract author and institution entities from citations"""

        entities = []

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            # Get citation authors
            cursor.execute("""
                SELECT DISTINCT authors FROM citations 
                WHERE paper_id = ? AND authors IS NOT NULL
            """, (paper_id,))

            for row in cursor.fetchall():
                authors_text = row[0]
                if authors_text:
                    # Split author names
                    authors = [name.strip() for name in authors_text.split(',')]

                    for author in authors:
                        if len(author) > 3:  # Skip short names
                            entity_id = self._generate_entity_id(paper_id, 'author', author.lower())

                            entity = Entity(
                                id=entity_id,
                                paper_id=paper_id,
                                entity_text=author,
                                entity_type='author',
                                normalized_form=author.lower(),
                                confidence_score=0.9,  # High confidence for citation authors
                                mention_count=1,
                                context_sentences=[],
                                aliases=[],
                                metadata={'source': 'citations'}
                            )

                            entities.append(entity)

        except Exception as e:
            self.logger.warning(f"Error extracting citation entities: {e}")

        return entities

    def _calculate_entity_confidence(self, entity_text: str, context: str, 
                                   context_keywords: List[str]) -> float:
        """Calculate confidence score for entity extraction"""

        base_confidence = 0.5

        # Boost confidence if surrounded by relevant context keywords
        context_lower = context.lower()
        keyword_boost = 0.0

        for keyword in context_keywords:
            if keyword in context_lower:
                keyword_boost += 0.1

        # Boost confidence for capitalized entities (likely proper nouns)
        if entity_text[0].isupper():
            base_confidence += 0.2

        # Boost confidence for multi-word entities
        if len(entity_text.split()) > 1:
            base_confidence += 0.1

        total_confidence = min(base_confidence + keyword_boost, 1.0)
        return total_confidence

    def _extract_sentence_context(self, text: str, position: int) -> str:
        """Extract sentence containing the entity mention"""

        # Find sentence boundaries around the position
        sentences = re.split(r'[.!?]+', text)

        current_pos = 0
        for sentence in sentences:
            if current_pos <= position <= current_pos + len(sentence):
                return sentence.strip()
            current_pos += len(sentence) + 1

        # Fallback: return surrounding 100 characters
        start = max(0, position - 50)
        end = min(len(text), position + 50)
        return text[start:end].strip()

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities and merge mentions"""

        unique_entities = {}

        for entity in entities:
            key = f"{entity.entity_type}_{entity.normalized_form}"

            if key in unique_entities:
                # Merge with existing entity
                existing = unique_entities[key]
                existing.mention_count += entity.mention_count
                existing.confidence_score = max(existing.confidence_score, entity.confidence_score)

                if entity.context_sentences:
                    existing.context_sentences.extend(entity.context_sentences)

            else:
                unique_entities[key] = entity

        return list(unique_entities.values())

    def _extract_relationships(self, paper_id: str, entities: List[Entity], 
                             chunks: List[Dict]) -> List[EntityRelationship]:
        """Extract relationships between entities"""

        relationships = []

        # Create entity lookup by normalized form for faster searching
        entity_by_normalized = {}
        for entity in entities:
            entity_by_normalized[entity.normalized_form] = entity

        # Extract relationships from chunks
        for chunk in chunks:
            chunk_text = chunk['text'].lower()  # Convert to lowercase for matching

            for rel_type, patterns in self.relationship_patterns.items():
                for pattern in patterns:
                    try:
                        matches = re.finditer(pattern, chunk_text, re.IGNORECASE)

                        for match in matches:
                            # Extract entities mentioned in the relationship context
                            context_text = match.group(0)

                            # Find entities that appear in this relationship context
                            entities_in_context = []
                            for entity in entities:
                                if entity.normalized_form in context_text.lower():
                                    entities_in_context.append(entity)

                            # Create relationships between pairs of entities
                            for i in range(len(entities_in_context)):
                                for j in range(i + 1, len(entities_in_context)):
                                    source_entity = entities_in_context[i]
                                    target_entity = entities_in_context[j]

                                    relationship_id = self._generate_relationship_id(
                                        paper_id, source_entity.id, target_entity.id, rel_type
                                    )

                                    relationship = EntityRelationship(
                                        id=relationship_id,
                                        paper_id=paper_id,
                                        source_entity_id=source_entity.id,
                                        target_entity_id=target_entity.id,
                                        relationship_type=rel_type,
                                        confidence_score=0.7,
                                        context=context_text,
                                        section_id=chunk['section_id'],
                                        relationship_strength=1.0
                                    )

                                    relationships.append(relationship)

                    except Exception as e:
                        self.logger.warning(f"Error processing relationship pattern {pattern}: {e}")
                        continue

        # Co-occurrence relationships: entities in same chunk
        chunk_entity_map: Dict[str, List[Entity]] = {}
        for entity in entities:
            if entity.chunk_id not in chunk_entity_map:
                chunk_entity_map[entity.chunk_id] = []
            chunk_entity_map[entity.chunk_id].append(entity)

        for chunk_id, ents in chunk_entity_map.items():
            for i in range(len(ents)):
                for j in range(i+1, len(ents)):
                    src, tgt = ents[i], ents[j]
                    rel_id = self._generate_relationship_id(paper_id, src.id, tgt.id, 'co_occurs_with')
                    rel = EntityRelationship(
                        id=rel_id,
                        paper_id=paper_id,
                        source_entity_id=src.id,
                        target_entity_id=tgt.id,
                        relationship_type='co_occurs_with',
                        confidence_score=0.5,           # moderate confidence
                        context='co-occurred in same section chunk',
                        section_id=src.section_id,
                        relationship_strength=1.0
                    )
                    relationships.append(rel)


        return relationships

    def _generate_entity_id(self, paper_id: str, entity_type: str, normalized_form: str) -> str:
        """Generate stable entity ID"""

        base = f"{paper_id}_{entity_type}_{normalized_form}"
        hash_suffix = hashlib.md5(base.encode()).hexdigest()[:8]
        return f"entity_{hash_suffix}"

    def _generate_relationship_id(self, paper_id: str, source_id: str, target_id: str, rel_type: str) -> str:
        """Generate stable relationship ID"""

        base = f"{paper_id}_{source_id}_{target_id}_{rel_type}"
        hash_suffix = hashlib.md5(base.encode()).hexdigest()[:8]
        return f"rel_{hash_suffix}"

    def _calculate_extraction_stats(self, paper_id: str, entities: List[Entity], 
                                  relationships: List[EntityRelationship]) -> EntityExtractionResult:
        """Calculate extraction statistics"""

        # Count entities by type
        entities_by_type = {}
        for entity in entities:
            entities_by_type[entity.entity_type] = entities_by_type.get(entity.entity_type, 0) + 1

        # Count relationships by type
        relationships_by_type = {}
        for rel in relationships:
            relationships_by_type[rel.relationship_type] = relationships_by_type.get(rel.relationship_type, 0) + 1

        # Calculate quality score
        quality_score = min(10.0, (len(entities) * 0.1) + (len(relationships) * 0.2))

        return EntityExtractionResult(
            paper_id=paper_id,
            total_entities=len(entities),
            entities_by_type=entities_by_type,
            total_relationships=len(relationships),
            relationships_by_type=relationships_by_type,
            processing_errors=[],
            quality_score=quality_score
        )

    def _save_entities_to_database(self, entities: List[Entity]):
        """Save entities to database"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            for entity in entities:
                cursor.execute("""
                    INSERT OR REPLACE INTO entities 
                    (id, paper_id, entity_text, entity_type, normalized_form, confidence_score,
                     section_id, chunk_id, first_mention_position, mention_count,
                     context_sentences, aliases, metadata_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.id,
                    entity.paper_id,
                    entity.entity_text,
                    entity.entity_type,
                    entity.normalized_form,
                    entity.confidence_score,
                    entity.section_id,
                    entity.chunk_id,
                    entity.first_mention_position,
                    entity.mention_count,
                    '|'.join(entity.context_sentences or []),
                    ','.join(entity.aliases or []),
                    str(entity.metadata) if entity.metadata else None,
                    datetime.now().isoformat()
                ))

            conn.commit()
            self.logger.info(f"Saved {len(entities)} entities to database")

        except Exception as e:
            self.logger.error(f"Error saving entities: {e}")

    def _save_relationships_to_database(self, relationships: List[EntityRelationship]):
        """Save relationships to database"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            for rel in relationships:
                cursor.execute("""
                    INSERT OR REPLACE INTO entity_relationships 
                    (id, paper_id, source_entity_id, target_entity_id, relationship_type,
                     confidence_score, context, section_id, relationship_strength, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rel.id,
                    rel.paper_id,
                    rel.source_entity_id,
                    rel.target_entity_id,
                    rel.relationship_type,
                    rel.confidence_score,
                    rel.context,
                    rel.section_id,
                    rel.relationship_strength,
                    datetime.now().isoformat()
                ))

            conn.commit()
            self.logger.info(f"Saved {len(relationships)} relationships to database")

        except Exception as e:
            self.logger.error(f"Error saving relationships: {e}")

    def _update_processing_status(self, paper_id: str, stage: str, success: bool):
        """Update processing status"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO processing_log (paper_id, processing_stage, status, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                paper_id,
                stage,
                'success' if success else 'failed',
                datetime.now().isoformat()
            ))

            conn.commit()
        except Exception as e:
            self.logger.error(f"Error updating processing status: {e}")

    def _create_empty_result(self, paper_id: str, error_msg: str) -> EntityExtractionResult:
        """Create empty result with error"""

        return EntityExtractionResult(
            paper_id=paper_id,
            total_entities=0,
            entities_by_type={},
            total_relationships=0,
            relationships_by_type={},
            processing_errors=[error_msg],
            quality_score=0.0
        )

    def get_extraction_stats(self, paper_id: str) -> Dict[str, Any]:
        """Get entity extraction statistics for a paper"""

        try:
            conn = self.db_manager.get_sqlite_connection()
            cursor = conn.cursor()

            # Get entity counts by type
            cursor.execute("""
                SELECT entity_type, COUNT(*), AVG(confidence_score), SUM(mention_count)
                FROM entities 
                WHERE paper_id = ? 
                GROUP BY entity_type
            """, (paper_id,))

            entity_stats = {}
            for row in cursor.fetchall():
                entity_type, count, avg_confidence, total_mentions = row
                entity_stats[entity_type] = {
                    'count': count,
                    'avg_confidence': round(avg_confidence or 0, 2),
                    'total_mentions': total_mentions or 0
                }

            # Get relationship counts by type
            cursor.execute("""
                SELECT relationship_type, COUNT(*), AVG(confidence_score)
                FROM entity_relationships 
                WHERE paper_id = ? 
                GROUP BY relationship_type
            """, (paper_id,))

            relationship_stats = {}
            for row in cursor.fetchall():
                rel_type, count, avg_confidence = row
                relationship_stats[rel_type] = {
                    'count': count,
                    'avg_confidence': round(avg_confidence or 0, 2)
                }

            return {
                'entities': entity_stats,
                'relationships': relationship_stats
            }

        except Exception as e:
            self.logger.error(f"Error getting extraction stats: {e}")
            return {}

# Standalone function for pipeline integration
def process_paper_entities(paper_id: str) -> EntityExtractionResult:
    """
    Extract entities and relationships for a paper.
    This is called after content chunking (Step 6).
    """
    extractor = EntityExtractor()
    return extractor.extract_entities_and_relationships(paper_id)



# test_entity_extractor.py - Simple test for entity extraction
from database_setup import DatabaseManager

def main():
    """Test entity extraction on a paper from the database"""

    # Initialize
    db_manager = DatabaseManager()
    extractor = EntityExtractor(db_manager)

    # Find a paper that has been processed through content chunking
    conn = db_manager.get_sqlite_connection()
    cursor = conn.cursor()

    # Get a paper that has content chunks
    cursor.execute("""
        SELECT DISTINCT paper_id 
        FROM content_chunks 
        LIMIT 1
    """)

    row = cursor.fetchone()
    if not row:
        print("No papers with content chunks found.")
        print("Run content chunking first!")
        return

    paper_id = row[0]
    print(f"Extracting entities for paper: {paper_id}")

    # Check available data
    cursor.execute("SELECT COUNT(*) FROM content_chunks WHERE paper_id = ?", (paper_id,))
    chunks_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM citations WHERE paper_id = ?", (paper_id,))
    citations_count = cursor.fetchone()[0]

    print(f"Available data:")
    print(f"  Content chunks: {chunks_count}")
    print(f"  Citations: {citations_count}")

    # Process entity extraction
    print("\nExtracting entities and relationships...")
    result = extractor.extract_entities_and_relationships(paper_id)

    # Display results
    print("\n" + "="*50)
    print("ENTITY EXTRACTION RESULTS")
    print("="*50)
    print(f"Paper ID: {result.paper_id}")
    print(f"Total entities extracted: {result.total_entities}")
    print(f"Total relationships found: {result.total_relationships}")
    print(f"Quality score: {result.quality_score:.2f}/10.0")

    if result.processing_errors:
        print(f"\nProcessing errors:")
        for error in result.processing_errors:
            print(f"  - {error}")

    # Show entities by type
    if result.entities_by_type:
        print(f"\nEntities by Type:")
        for entity_type, count in result.entities_by_type.items():
            print(f"  {entity_type}: {count}")

    # Show relationships by type
    if result.relationships_by_type:
        print(f"\nRelationships by Type:")
        for rel_type, count in result.relationships_by_type.items():
            print(f"  {rel_type}: {count}")

    # Show detailed statistics
    stats = extractor.get_extraction_stats(paper_id)
    if stats:
        print(f"\nDetailed Entity Statistics:")
        for entity_type, type_stats in stats.get('entities', {}).items():
            print(f"  {entity_type}: {type_stats['count']} entities, "
                  f"avg confidence: {type_stats['avg_confidence']}, "
                  f"total mentions: {type_stats['total_mentions']}")

        if stats.get('relationships'):
            print(f"\nDetailed Relationship Statistics:")
            for rel_type, rel_stats in stats.get('relationships', {}).items():
                print(f"  {rel_type}: {rel_stats['count']} relationships, "
                      f"avg confidence: {rel_stats['avg_confidence']}")

    # Show example entities
    cursor.execute("""
        SELECT entity_text, entity_type, confidence_score, mention_count, normalized_form
        FROM entities 
        WHERE paper_id = ? 
        ORDER BY confidence_score DESC, mention_count DESC
        LIMIT 10
    """, (paper_id,))

    entities = cursor.fetchall()
    if entities:
        print(f"\nTop Entities (by confidence and mentions):")
        for entity_text, entity_type, confidence, mentions, normalized in entities:
            print(f"  '{entity_text}' ({entity_type})")
            print(f"    Confidence: {confidence:.2f}, Mentions: {mentions}")
            print(f"    Normalized: {normalized}")
            print()

    # Show example relationships
    cursor.execute("""
        SELECT er.relationship_type, e1.entity_text, e2.entity_text, er.confidence_score, er.context
        FROM entity_relationships er
        JOIN entities e1 ON er.source_entity_id = e1.id
        JOIN entities e2 ON er.target_entity_id = e2.id
        WHERE er.paper_id = ?
        ORDER BY er.confidence_score DESC
        LIMIT 5
    """, (paper_id,))

    relationships = cursor.fetchall()
    if relationships:
        print(f"Example Relationships:")
        for rel_type, source_entity, target_entity, confidence, context in relationships:
            print(f"  {source_entity} --[{rel_type}]--> {target_entity}")
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Context: {context[:60]}{'...' if len(context) > 60 else ''}")
            print()

    # Show entities by section
    cursor.execute("""
        SELECT e.entity_type, cc.canonical_section, COUNT(*) as count
        FROM entities e
        JOIN content_chunks cc ON e.chunk_id = cc.id
        WHERE e.paper_id = ?
        GROUP BY e.entity_type, cc.canonical_section
        ORDER BY count DESC
    """, (paper_id,))

    section_dist = cursor.fetchall()
    if section_dist:
        print(f"Entity Distribution by Section:")
        current_type = None
        for entity_type, section, count in section_dist:
            if entity_type != current_type:
                print(f"\n  {entity_type}:")
                current_type = entity_type
            print(f"    {section}: {count}")

    # Show high-confidence entities for graph database
    cursor.execute("""
        SELECT entity_text, entity_type, confidence_score, mention_count
        FROM entities 
        WHERE paper_id = ? AND confidence_score > 0.7
        ORDER BY entity_type, confidence_score DESC
    """, (paper_id,))

    high_conf_entities = cursor.fetchall()
    if high_conf_entities:
        print(f"\nHigh-Confidence Entities (for Graph Database):")
        current_type = None
        for entity_text, entity_type, confidence, mentions in high_conf_entities:
            if entity_type != current_type:
                print(f"\n  {entity_type.upper()}:")
                current_type = entity_type
            print(f"    {entity_text} (conf: {confidence:.2f}, mentions: {mentions})")

    print("\n" + "="*50)
    print("✅ Entity extraction complete!")
    print("Entities and relationships are ready for graph database storage.")
    print("Check 'entities' and 'entity_relationships' tables.")
    print("="*50)

if __name__ == "__main__":
    main()

# %%
