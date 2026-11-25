//! Read a source into bytes and a media type.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use reqwest::redirect::Policy;

use crate::error::{Error, Result};

/// Default extraction prompt sent to general structured-output providers.
pub const PROMPT: &str = "Extract the requested information from this file.";

/// Office, PDF, and OpenDocument types mapped to their canonical extensions.
pub const DOCUMENT_EXTENSIONS: &[(&str, &str)] = &[
    ("application/pdf", ".pdf"),
    ("application/rtf", ".rtf"),
    ("application/msword", ".doc"),
    ("application/vnd.ms-excel", ".xls"),
    ("application/vnd.ms-powerpoint", ".ppt"),
    ("application/vnd.oasis.opendocument.presentation", ".odp"),
    ("application/vnd.oasis.opendocument.spreadsheet", ".ods"),
    ("application/vnd.oasis.opendocument.text", ".odt"),
    (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
    (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
        ".dotx",
    ),
];

const TEXT_TYPES: &[&str] = &[
    "application/json",
    "application/xml",
    "application/javascript",
];
const GENERIC_TYPES: &[&str] = &["", "application/octet-stream", "binary/octet-stream"];
const URL_TIMEOUT: Duration = Duration::from_secs(30);

/// Input source for extraction.
///
/// Converting a `&str` or `String` produces `Source::Url` when it starts with
/// `http://` or `https://`, and `Source::Text` otherwise (it never reads a local
/// file). Use `Source::Path` (or pass a `PathBuf` / `&Path`) for local files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    Text(String),
    Path(PathBuf),
    Url(String),
    Bytes(Vec<u8>),
}

impl From<&str> for Source {
    fn from(s: &str) -> Self {
        if is_http_url(s) {
            Self::Url(s.to_string())
        } else {
            Self::Text(s.to_string())
        }
    }
}

impl From<String> for Source {
    fn from(s: String) -> Self {
        if is_http_url(&s) {
            Self::Url(s)
        } else {
            Self::Text(s)
        }
    }
}

impl From<PathBuf> for Source {
    fn from(path: PathBuf) -> Self {
        Self::Path(path)
    }
}

impl From<&Path> for Source {
    fn from(path: &Path) -> Self {
        Self::Path(path.to_path_buf())
    }
}

impl From<Vec<u8>> for Source {
    fn from(bytes: Vec<u8>) -> Self {
        Self::Bytes(bytes)
    }
}

impl From<&[u8]> for Source {
    fn from(bytes: &[u8]) -> Self {
        Self::Bytes(bytes.to_vec())
    }
}

/// A loaded source and the encodings providers ask for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Content {
    pub data: Vec<u8>,
    pub media_type: String,
}

impl Content {
    pub fn base64(&self) -> String {
        BASE64.encode(&self.data)
    }

    pub fn data_uri(&self) -> String {
        format!("data:{};base64,{}", self.media_type, self.base64())
    }

    pub fn is_text(&self) -> bool {
        self.media_type.starts_with("text/") || TEXT_TYPES.contains(&self.media_type.as_str())
    }

    pub fn is_image(&self) -> bool {
        self.media_type.starts_with("image/")
    }

    pub fn is_pdf(&self) -> bool {
        self.media_type == "application/pdf"
    }

    pub fn text(&self) -> Cow<'_, str> {
        String::from_utf8_lossy(&self.data)
    }
}

/// Look up the canonical file extension for a known document media type.
pub fn document_extension(media_type: &str) -> Option<&'static str> {
    DOCUMENT_EXTENSIONS
        .iter()
        .find_map(|(m, ext)| (*m == media_type).then_some(*ext))
}

/// Read text, a URL, bytes, or a local path into [`Content`]. An explicit `media_type` always wins.
///
/// Source URLs are downloaded with a fresh HTTP client (30s timeout, following redirects)
/// so model credentials are never sent to source hosts.
pub async fn load(source: Source, media_type: Option<&str>) -> Result<Content> {
    match source {
        Source::Url(url) => {
            let client = reqwest::Client::builder()
                .timeout(URL_TIMEOUT)
                .redirect(Policy::limited(10))
                .build()
                .map_err(|e| fetch_error(e, None))?;
            let response = client
                .get(&url)
                .send()
                .await
                .map_err(|e| fetch_error(e, None))?;
            let status = response.status();
            if !status.is_success() {
                let err = response.error_for_status().unwrap_err();
                return Err(fetch_error(err, Some(status.as_u16())));
            }
            let final_path = response.url().path().to_string();
            let header_type = response
                .headers()
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|v| {
                    v.split(';')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .to_ascii_lowercase()
                })
                .filter(|h| !GENERIC_TYPES.contains(&h.as_str()));
            let bytes = response
                .bytes()
                .await
                .map_err(|e| fetch_error(e, None))?
                .to_vec();
            build_content(bytes, media_type, &final_path, header_type.as_deref())
        }
        Source::Bytes(bytes) => build_content(bytes, media_type, "", None),
        Source::Text(text) => build_content(
            text.into_bytes(),
            Some(media_type.unwrap_or("text/plain")),
            "",
            None,
        ),
        Source::Path(path) => {
            let bytes = std::fs::read(&path)?;
            let name = path.to_string_lossy();
            build_content(bytes, media_type, &name, None)
        }
    }
}

fn is_http_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

fn build_content(
    data: Vec<u8>,
    media_type: Option<&str>,
    name: &str,
    header: Option<&str>,
) -> Result<Content> {
    let resolved = media_type
        .map(str::to_string)
        .or_else(|| type_from_name(name))
        .or_else(|| header.map(str::to_string))
        .or_else(|| sniff(&data).map(str::to_string))
        .ok_or_else(|| {
            Error::InvalidConfig(
                "Could not infer the media type; pass media_type explicitly.".to_string(),
            )
        })?;
    Ok(Content {
        data,
        media_type: resolved,
    })
}

fn type_from_name(name: &str) -> Option<String> {
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| format!(".{}", e.to_ascii_lowercase()))?;
    if let Some((media, _)) = DOCUMENT_EXTENSIONS.iter().find(|(_, e)| *e == ext) {
        return Some((*media).to_string());
    }
    mime_guess::from_ext(ext.trim_start_matches('.'))
        .first_raw()
        .map(str::to_string)
}

/// Guess a media type from leading bytes, falling back to `text/plain` for valid UTF-8.
pub fn sniff(data: &[u8]) -> Option<&'static str> {
    const SIGNATURES: &[(&[u8], &str)] = &[
        (b"%PDF", "application/pdf"),
        (b"\x89PNG", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF8", "image/gif"),
        (b"ID3", "audio/mpeg"),
        (b"fLaC", "audio/flac"),
        (b"OggS", "audio/ogg"),
    ];
    for (sig, media) in SIGNATURES {
        if data.starts_with(sig) {
            return Some(media);
        }
    }
    if data.starts_with(b"RIFF") {
        return match data.get(8..12) {
            Some(b"WEBP") => Some("image/webp"),
            Some(b"WAVE") => Some("audio/wav"),
            _ => None,
        };
    }
    if data.get(4..8) == Some(b"ftyp") {
        return Some("video/mp4");
    }
    if std::str::from_utf8(data).is_ok() {
        return Some("text/plain");
    }
    None
}

fn fetch_error(err: reqwest::Error, status: Option<u16>) -> Error {
    let detail = if let Some(code) = status {
        format!("HTTP {code}")
    } else if err.is_timeout() {
        "TimeoutError".to_string()
    } else if err.is_connect() {
        "ConnectError".to_string()
    } else if err.is_redirect() {
        "TooManyRedirects".to_string()
    } else if err.is_decode() {
        "DecodingError".to_string()
    } else {
        "RequestError".to_string()
    };
    // Never include the URL (which may contain secret query tokens) in the user-facing message.
    Error::extraction_with_source(
        format!(
            "Could not fetch the source URL ({detail}).\n\
             Next step: Check the URL and network access. If the source needs authentication \
             or a custom timeout, download it with your own HTTP client \
             and pass its bytes to extract()."
        ),
        err,
    )
}
