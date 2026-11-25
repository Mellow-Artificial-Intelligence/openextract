//! Read a source into bytes and a media type.

use std::borrow::Cow;
use std::fmt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use reqwest::redirect::Policy;
use url::Url;

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
/// Largest source-URL response that will be buffered.
///
/// `Content-Length` is rejected before the body is read when it is larger.
/// The body is always streamed, so a missing or understated length cannot
/// exceed this either.
pub const MAX_DOWNLOAD_BYTES: u64 = 50 * 1024 * 1024;
/// Same-host redirects followed after the original source request.
const MAX_SOURCE_REDIRECTS: usize = 3;

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
/// Source URLs are downloaded with a fresh HTTP client (30s timeout) so model
/// credentials are never sent to source hosts. At most 3 redirects are followed,
/// and only to the same host and port (an `http` to `https` upgrade on the
/// default ports is allowed). Responses larger than [`MAX_DOWNLOAD_BYTES`] are rejected from
/// `Content-Length` when that header is present, and by a streaming cap in every case.
pub async fn load(source: Source, media_type: Option<&str>) -> Result<Content> {
    match source {
        Source::Url(url) => load_url(&url, media_type, MAX_DOWNLOAD_BYTES).await,
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

async fn load_url(url: &str, media_type: Option<&str>, max_bytes: u64) -> Result<Content> {
    let client = reqwest::Client::builder()
        .timeout(URL_TIMEOUT)
        .redirect(source_redirect_policy())
        .build()
        .map_err(|e| fetch_error(e, None))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| fetch_error(e, None))?;
    let status = response.status();
    if !status.is_success() {
        // 3xx is not an `error_for_status` error. Reqwest leaves a redirect
        // unfollowed when the Location is not an HTTP(S) URL it can resolve.
        if status.is_redirection() {
            return Err(unfollowed_redirect(&response));
        }
        return Err(match response.error_for_status() {
            Err(err) => fetch_error(err, Some(status.as_u16())),
            Ok(_) => Error::extraction(fetch_message(&format!("HTTP {}", status.as_u16()))),
        });
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
    let bytes = read_capped_body(response, max_bytes).await?;
    build_content(bytes, media_type, &final_path, header_type.as_deref())
}

/// Why a redirect target is refused, if it is.
fn redirect_block_reason(original: &Url, next: &Url) -> Option<&'static str> {
    if next.scheme() != "http" && next.scheme() != "https" {
        return Some("redirect target must use http or https");
    }
    if !next.username().is_empty() || next.password().is_some() {
        return Some("redirect target must not include credentials");
    }
    let Some(original_host) = original.host_str() else {
        return Some("redirect target must stay on the original host");
    };
    let Some(next_host) = next.host_str() else {
        return Some("redirect target must stay on the original host");
    };
    if !original_host.eq_ignore_ascii_case(next_host) {
        return Some("redirect target must stay on the original host");
    }
    match (original.scheme(), next.scheme()) {
        ("https", "http") => Some("redirect must not downgrade from https"),
        ("http", "http") | ("https", "https") => {
            if original.port_or_known_default() != next.port_or_known_default() {
                Some("redirect target must stay on the original port")
            } else {
                None
            }
        }
        ("http", "https") => {
            let default_upgrade = original.port_or_known_default() == Some(80)
                && next.port_or_known_default() == Some(443);
            let same_port = original.port_or_known_default() == next.port_or_known_default();
            if default_upgrade || same_port {
                None
            } else {
                Some("redirect target must stay on the original port")
            }
        }
        _ => Some("redirect target must use http or https"),
    }
}

fn source_redirect_policy() -> Policy {
    Policy::custom(|attempt| {
        // `previous` includes the URL just requested, so its length is one more
        // than the number of redirects already followed.
        if attempt.previous().len() > MAX_SOURCE_REDIRECTS {
            return attempt.error(SourceRedirect("too many redirects"));
        }
        if attempt
            .previous()
            .iter()
            .any(|previous| previous == attempt.url())
        {
            return attempt.error(SourceRedirect("redirect loop"));
        }
        if let Some(original) = attempt.previous().first()
            && let Some(reason) = redirect_block_reason(original, attempt.url())
        {
            return attempt.error(SourceRedirect(reason));
        }
        attempt.follow()
    })
}

#[derive(Debug)]
struct SourceRedirect(&'static str);

impl fmt::Display for SourceRedirect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for SourceRedirect {}

fn content_length_over_limit(response: &reqwest::Response, max_bytes: u64) -> bool {
    if let Some(raw) = response
        .headers()
        .get(reqwest::header::CONTENT_LENGTH)
        .and_then(|value| value.to_str().ok())
    {
        let raw = raw.trim();
        if let Ok(len) = raw.parse::<u64>() {
            return len > max_bytes;
        }
        // A digit string that does not fit in u64 is larger than any cap we use.
        if !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit()) {
            return true;
        }
    }
    response.content_length().is_some_and(|len| len > max_bytes)
}

async fn read_capped_body(mut response: reqwest::Response, max_bytes: u64) -> Result<Vec<u8>> {
    if content_length_over_limit(&response, max_bytes) {
        return Err(download_too_large(max_bytes));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|err| fetch_error(err, None))?
    {
        if chunk.len() as u64 > max_bytes.saturating_sub(body.len() as u64) {
            return Err(download_too_large(max_bytes));
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn download_too_large(max_bytes: u64) -> Error {
    Error::extraction(format!(
        "Could not fetch the source URL (response exceeds the {} download limit).\n\
         Next step: Download a smaller file, or fetch the URL with your own HTTP client \
         and pass its bytes to extract().",
        limit_label(max_bytes)
    ))
}

fn limit_label(max_bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    if max_bytes >= MIB && max_bytes.is_multiple_of(MIB) {
        format!("{} MiB", max_bytes / MIB)
    } else {
        format!("{max_bytes} bytes")
    }
}

fn unfollowed_redirect(response: &reqwest::Response) -> Error {
    let detail = response
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| {
            response
                .url()
                .join(raw)
                .ok()
                .or_else(|| Url::parse(raw).ok())
        })
        .and_then(|next| redirect_block_reason(response.url(), &next))
        .unwrap_or("redirect was not allowed");
    Error::extraction(fetch_message(detail))
}

fn fetch_message(detail: &str) -> String {
    // Never include the URL (which may contain secret query tokens) in the user-facing message.
    format!(
        "Could not fetch the source URL ({detail}).\n\
         Next step: Check the URL and network access. If the source needs authentication \
         or a custom timeout, download it with your own HTTP client \
         and pass its bytes to extract()."
    )
}

fn fetch_error(err: reqwest::Error, status: Option<u16>) -> Error {
    let detail = if let Some(code) = status {
        format!("HTTP {code}")
    } else if err.is_timeout() {
        "TimeoutError".to_string()
    } else if err.is_connect() {
        "ConnectError".to_string()
    } else if err.is_redirect() {
        redirect_detail(&err).to_string()
    } else if err.is_decode() {
        "DecodingError".to_string()
    } else {
        "RequestError".to_string()
    };
    Error::extraction_with_source(fetch_message(&detail), err)
}

fn redirect_detail(err: &reqwest::Error) -> &str {
    std::error::Error::source(err)
        .and_then(|source| source.downcast_ref::<SourceRedirect>())
        .map(|rejected| rejected.0)
        .unwrap_or("redirect was not allowed")
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::time::Duration;

    use super::*;

    fn serve(response: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let mut buf = [0u8; 2048];
            let _ = sock.read(&mut buf);
            let _ = sock.write_all(&response);
            let _ = sock.read(&mut buf);
        });
        format!("http://{addr}/file.txt")
    }

    fn chunked(body: &[u8]) -> Vec<u8> {
        let mut response = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\
            Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
            .to_vec();
        for chunk in body.chunks(4) {
            response.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
            response.extend_from_slice(chunk);
            response.extend_from_slice(b"\r\n");
        }
        response.extend_from_slice(b"0\r\n\r\n");
        response
    }

    #[test]
    fn redirect_targets_stay_on_the_original_host_and_port() {
        let allowed = [
            ("http://example.com/a", "https://example.com/b"),
            ("http://example.com/a", "http://example.com/b"),
            ("https://example.com/a", "https://example.com/b"),
            ("http://example.com:8080/a", "http://example.com:8080/b"),
            ("http://example.com:8443/a", "https://example.com:8443/b"),
            ("http://Example.COM/a", "https://example.com/b"),
            ("http://127.0.0.1:9/a", "http://127.0.0.1:9/b"),
        ];
        for (from, to) in allowed {
            let reason =
                redirect_block_reason(&Url::parse(from).unwrap(), &Url::parse(to).unwrap());
            assert!(reason.is_none(), "{from} -> {to} blocked: {reason:?}");
        }

        let blocked = [
            ("https://example.com/a", "http://example.com/b", "downgrade"),
            (
                "http://example.com/a",
                "http://evil.test/b",
                "original host",
            ),
            (
                "http://example.com/a",
                "http://example.com:8080/b",
                "original port",
            ),
            (
                "http://example.com/a",
                "https://user:pw@example.com/b",
                "credentials",
            ),
            (
                "http://example.com/a",
                "file:///etc/passwd",
                "http or https",
            ),
            (
                "http://example.com/a",
                "http://example.com.evil.test/b",
                "original host",
            ),
        ];
        for (from, to, needle) in blocked {
            let reason =
                redirect_block_reason(&Url::parse(from).unwrap(), &Url::parse(to).unwrap())
                    .unwrap_or("allowed");
            assert!(reason.contains(needle), "{from} -> {to} yielded {reason}");
        }
    }

    #[tokio::test]
    async fn chunked_body_is_capped_without_content_length() {
        let body = b"0123456789abcdef";
        let err = load_url(&serve(chunked(body)), None, 8).await.unwrap_err();
        let message = err.to_string();
        assert!(message.contains("8 bytes download limit"), "{message}");
        assert!(message.contains("Next step:"));
        assert!(matches!(err, Error::Extraction { .. }));

        let content = load_url(&serve(chunked(body)), None, 16).await.unwrap();
        assert_eq!(content.data, body);
        assert_eq!(content.media_type, "text/plain");
    }

    #[tokio::test]
    async fn declared_content_length_is_enforced_before_the_body_is_read() {
        let ok = serve(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\
                Connection: close\r\n\r\nhello"
                .to_vec(),
        );
        let content = load_url(&ok, None, 5).await.unwrap();
        assert_eq!(content.data, b"hello");

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            let (mut sock, _) = listener.accept().unwrap();
            sock.set_read_timeout(Some(Duration::from_secs(5))).ok();
            let mut buf = [0u8; 2048];
            let _ = sock.read(&mut buf);
            let _ = sock.write_all(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 1000000\r\n\
                    Connection: close\r\n\r\n",
            );
            // The body is never sent. Reading it would block until this timeout.
            let _ = sock.read(&mut buf);
        });
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            load_url(&format!("http://{addr}/huge.bin"), None, 32),
        )
        .await;
        let err = result
            .expect("Content-Length was not rejected before the body")
            .unwrap_err();
        assert!(err.to_string().contains("32 bytes download limit"));
    }
}
