"""Email loader for EML, MSG, MBOX files."""

import email
import logging
import mailbox
from email.header import decode_header
from pathlib import Path

from langchain.schema import Document


class EmailLoader:
    """Loader for email formats (EML, MSG, MBOX)."""

    def __init__(self, file_path: str, logger: logging.Logger | None = None):
        """Initialize the email loader.

        Args:
            file_path: Path to the email file
            logger: Optional logger instance
        """
        self.file_path = Path(file_path)
        self.logger = logger or logging.getLogger(__name__)

        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.file_type = self.file_path.suffix.lower()

        # Validate supported formats
        supported_formats = ['.eml', '.msg', '.mbox']
        if self.file_type not in supported_formats:
            raise ValueError(f"Unsupported email format: {self.file_type}")

    def load(self) -> list[Document]:
        """Load the email file and return Document objects.

        Returns:
            List of Document objects with extracted content

        Raises:
            Exception: If email loading fails
        """
        try:
            if self.file_type == '.eml':
                return self._load_eml()
            elif self.file_type == '.msg':
                return self._load_msg()
            elif self.file_type == '.mbox':
                return self._load_mbox()
            else:
                raise ValueError(f"Unsupported file type: {self.file_type}")

        except Exception as e:
            self.logger.error(f"Failed to load {self.file_path}: {e}")
            raise

    def _load_eml(self) -> list[Document]:
        """Load EML email file.

        Returns:
            List of Document objects
        """
        try:
            with open(self.file_path, 'rb') as file:
                msg = email.message_from_bytes(file.read())

            return [self._process_email_message(msg)]

        except Exception as e:
            self.logger.error(f"Failed to load EML: {e}")
            raise

    def _load_msg(self) -> list[Document]:
        """Load MSG email file (Outlook format).

        Returns:
            List of Document objects
        """
        try:
            # Try to use extract_msg library for MSG files
            try:
                import extract_msg
                msg = extract_msg.Message(str(self.file_path))

                # Extract content
                content_parts = []

                # Basic email info
                if msg.subject:
                    content_parts.append(f"Subject: {msg.subject}")
                if msg.sender:
                    content_parts.append(f"From: {msg.sender}")
                if msg.to:
                    content_parts.append(f"To: {msg.to}")
                if msg.cc:
                    content_parts.append(f"CC: {msg.cc}")
                if msg.date:
                    content_parts.append(f"Date: {msg.date}")

                content_parts.append("")  # Empty line

                # Email body
                if msg.body:
                    content_parts.append("Body:")
                    content_parts.append(msg.body)

                # Attachments info
                if msg.attachments:
                    content_parts.append(f"\nAttachments ({len(msg.attachments)}):")
                    for attachment in msg.attachments:
                        content_parts.append(f"- {attachment.longFilename or attachment.shortFilename}")

                full_content = "\n".join(content_parts)

                metadata = {
                    'source': str(self.file_path.resolve()),
                    'file_type': self.file_type,
                    'loader': 'EmailLoader',
                    'subject': msg.subject or "",
                    'sender': msg.sender or "",
                    'recipients': msg.to or "",
                    'date': str(msg.date) if msg.date else "",
                    'attachment_count': len(msg.attachments) if msg.attachments else 0
                }

                msg.close()

                return [Document(page_content=full_content, metadata=metadata)]

            except ImportError:
                self.logger.warning("extract_msg not available, treating MSG as binary")
                # Fallback: read as binary and try to extract what we can
                with open(self.file_path, 'rb') as file:
                    content = file.read()

                # Try to find readable text
                try:
                    text_content = content.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    text_content = str(content)

                metadata = {
                    'source': str(self.file_path.resolve()),
                    'file_type': self.file_type,
                    'loader': 'EmailLoader',
                    'note': 'Parsed as binary due to missing extract_msg library'
                }

                return [Document(page_content=text_content, metadata=metadata)]

        except Exception as e:
            self.logger.error(f"Failed to load MSG: {e}")
            raise

    def _load_mbox(self) -> list[Document]:
        """Load MBOX mailbox file.

        Returns:
            List of Document objects (one per email)
        """
        try:
            mbox = mailbox.mbox(str(self.file_path))
            documents = []

            for i, message in enumerate(mbox):
                try:
                    doc = self._process_email_message(message)
                    # Add mailbox-specific metadata
                    doc.metadata['mailbox_index'] = i
                    doc.metadata['total_messages'] = len(mbox)
                    documents.append(doc)
                except Exception as e:
                    self.logger.warning(f"Failed to process message {i} in mbox: {e}")
                    continue

            self.logger.info(f"Loaded {len(documents)} emails from MBOX")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to load MBOX: {e}")
            raise

    def _process_email_message(self, msg: email.message.Message) -> Document:
        """Process an email message and create a Document.

        Args:
            msg: Email message object

        Returns:
            Document object with email content
        """
        content_parts = []

        # Extract headers
        subject = self._decode_header(msg.get('Subject', ''))
        from_addr = self._decode_header(msg.get('From', ''))
        to_addr = self._decode_header(msg.get('To', ''))
        cc_addr = self._decode_header(msg.get('CC', ''))
        date = msg.get('Date', '')

        # Add header information
        if subject:
            content_parts.append(f"Subject: {subject}")
        if from_addr:
            content_parts.append(f"From: {from_addr}")
        if to_addr:
            content_parts.append(f"To: {to_addr}")
        if cc_addr:
            content_parts.append(f"CC: {cc_addr}")
        if date:
            content_parts.append(f"Date: {date}")

        content_parts.append("")  # Empty line

        # Extract body
        body = self._extract_email_body(msg)
        if body:
            content_parts.append("Body:")
            content_parts.append(body)

        # Extract attachments info
        attachments = self._extract_attachment_info(msg)
        if attachments:
            content_parts.append(f"\nAttachments ({len(attachments)}):")
            for attachment in attachments:
                content_parts.append(f"- {attachment}")

        full_content = "\n".join(content_parts)

        # Create metadata
        metadata = {
            'source': str(self.file_path.resolve()),
            'file_type': self.file_type,
            'loader': 'EmailLoader',
            'subject': subject,
            'from': from_addr,
            'to': to_addr,
            'cc': cc_addr,
            'date': date,
            'attachment_count': len(attachments)
        }

        return Document(page_content=full_content, metadata=metadata)

    def _decode_header(self, header_value: str) -> str:
        """Decode email header value.

        Args:
            header_value: Raw header value

        Returns:
            Decoded header string
        """
        if not header_value:
            return ""

        try:
            decoded_parts = decode_header(header_value)
            decoded_string = ""

            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        decoded_string += part.decode(encoding)
                    else:
                        decoded_string += part.decode('utf-8', errors='ignore')
                else:
                    decoded_string += part

            return decoded_string
        except Exception:
            return header_value

    def _extract_email_body(self, msg: email.message.Message) -> str:
        """Extract email body content.

        Args:
            msg: Email message object

        Returns:
            Email body text
        """
        body_parts = []

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))

                # Skip attachments
                if 'attachment' in content_disposition:
                    continue

                if content_type == 'text/plain':
                    try:
                        body_parts.append(part.get_payload(decode=True).decode('utf-8', errors='ignore'))
                    except:
                        continue
                elif content_type == 'text/html':
                    try:
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        # Simple HTML to text conversion
                        import re
                        text_content = re.sub(r'<[^>]+>', '', html_content)
                        body_parts.append(text_content)
                    except (UnicodeDecodeError, AttributeError):
                        continue
        else:
            # Single part message
            try:
                body_parts.append(msg.get_payload(decode=True).decode('utf-8', errors='ignore'))
            except (UnicodeDecodeError, AttributeError):
                body_parts.append(str(msg.get_payload()))

        return "\n\n".join(body_parts)

    def _extract_attachment_info(self, msg: email.message.Message) -> list[str]:
        """Extract attachment information.

        Args:
            msg: Email message object

        Returns:
            List of attachment names
        """
        attachments = []

        if msg.is_multipart():
            for part in msg.walk():
                content_disposition = str(part.get('Content-Disposition', ''))

                if 'attachment' in content_disposition:
                    filename = part.get_filename()
                    if filename:
                        attachments.append(self._decode_header(filename))
                    else:
                        attachments.append("unnamed_attachment")

        return attachments
