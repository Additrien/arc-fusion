#!/usr/bin/env python3
"""
Admin script for batch document operations and database maintenance.
"""
import os
import sys
import asyncio
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStore

class IngestAdmin:
    """Administrative utilities for document ingestion and maintenance."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
    
    async def batch_ingest(self, directory: str):
        """Ingest all PDF files from a directory."""
        pdf_files = list(Path(directory).glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to ingest...")
        
        for pdf_file in pdf_files:
            try:
                print(f"Processing {pdf_file.name}...")
                
                with open(pdf_file, 'rb') as f:
                    content = f.read()
                
                result = await self.document_processor.process_document(
                    content, pdf_file.name
                )
                
                await self.vector_store.store_document_chunks(result)
                
                print(f"✓ {pdf_file.name}: {result['parent_chunk_count']} parent chunks, "
                      f"{result['child_chunk_count']} child chunks")
                
            except Exception as e:
                print(f"✗ Failed to process {pdf_file.name}: {str(e)}")
    
    async def show_stats(self):
        """Display database statistics."""
        try:
            stats = await self.vector_store.get_database_stats()
            documents = await self.vector_store.get_all_documents()
            
            print("\n=== Database Statistics ===")
            print(f"Total chunks: {stats['total_chunks']}")
            print(f"Unique documents: {stats['unique_documents']}")
            print(f"Collection: {stats['collection_name']}")
            print(f"Weaviate URL: {stats['weaviate_url']}")
            
            print("\n=== Documents ===")
            for doc in documents[:10]:  # Show first 10
                print(f"- {doc['filename']} ({doc['document_id']})")
            
            if len(documents) > 10:
                print(f"... and {len(documents) - 10} more")
                
        except Exception as e:
            print(f"Failed to get stats: {str(e)}")
    
    async def clear_database(self):
        """Clear all documents from the database."""
        try:
            confirmation = input("Are you sure you want to clear ALL documents? (yes/no): ")
            if confirmation.lower() != 'yes':
                print("Operation cancelled.")
                return
            
            await self.vector_store.clear_all_documents()
            await self.document_processor.clear_parent_store()
            
            print("✓ Database cleared successfully.")
            
        except Exception as e:
            print(f"Failed to clear database: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description="Arc-Fusion Admin Utilities")
    parser.add_argument('command', choices=['ingest', 'stats', 'clear'], 
                       help='Command to execute')
    parser.add_argument('--directory', '-d', 
                       help='Directory containing PDF files (for ingest command)')
    
    args = parser.parse_args()
    
    admin = IngestAdmin()
    
    # Wait a moment for services to initialize
    await asyncio.sleep(2)
    
    if args.command == 'ingest':
        if not args.directory:
            print("Error: --directory required for ingest command")
            sys.exit(1)
        await admin.batch_ingest(args.directory)
    
    elif args.command == 'stats':
        await admin.show_stats()
    
    elif args.command == 'clear':
        await admin.clear_database()

if __name__ == "__main__":
    asyncio.run(main()) 