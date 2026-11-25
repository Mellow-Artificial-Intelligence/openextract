mod common;

use std::path::PathBuf;

use openextract::content::{DOCUMENT_EXTENSIONS, load, sniff};
use openextract::{Error, ExtractOptions, Source, extract, extract_with_options};
use serde_json::{Value, json};
use tempfile::tempdir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use common::{Invoice, configured, mount_json, output, recorded_bodies};

const PDF_BYTES: &[u8] = b"%PDF-1.7 fake";

fn pdf_file_json() -> Value {
    json!({
        "type": "input_file",
        "filename": "document.pdf",
        "file_data": "data:application/pdf;base64,JVBERi0xLjcgZmFrZQ==",
    })
}

fn last_document(bodies: &[Value]) -> &Value {
    bodies.last().unwrap()["input"][0]["content"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
}

#[tokio::test]
async fn test_strings_are_text_and_never_read_local_files() {
    let server = MockServer::start().await;
    mount_json(&server, 200, output("responses", None)).await;

    let long_text = "x".repeat(10_000);
    let cases = [
        "Total due: 120.00",
        "Vendor: Café 東京\nTotal due: 120.00\n",
        "%PDF is literal text here",
        r#"{"total": 120.00}"#,
        long_text.as_str(),
        "",
        "Cargo.toml", // exists on disk in rust/ — must still be treated as literal text!
    ];

    for text in cases {
        let inv: Invoice = extract(configured(&server, "openai:test"), text, None)
            .await
            .unwrap();
        assert_eq!(inv.total, 42.0);
        let bodies = recorded_bodies(&server).await;
        assert_eq!(
            last_document(&bodies),
            &json!({"type": "input_text", "text": text})
        );
    }

    let content = load(Source::from("Total due: 120.00"), None).await.unwrap();
    assert_eq!(content.media_type, "text/plain");
}

#[tokio::test]
async fn test_path_reading_missing_file_and_office_extensions() {
    let server = MockServer::start().await;
    mount_json(&server, 200, output("responses", None)).await;
    let dir = tempdir().unwrap();

    let pdf_path = dir.path().join("invoice.pdf");
    std::fs::write(&pdf_path, PDF_BYTES).unwrap();
    let _: Invoice = extract(configured(&server, "openai:test"), pdf_path.as_path(), None)
        .await
        .unwrap();
    assert_eq!(
        last_document(&recorded_bodies(&server).await),
        &pdf_file_json()
    );

    let missing = PathBuf::from("/missing/does-not-exist-invoice.pdf");
    let err = extract::<Invoice>(configured(&server, "openai:test"), missing, None)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Io(e) if e.kind() == std::io::ErrorKind::NotFound));

    for (media, ext) in DOCUMENT_EXTENSIONS {
        let p = dir.path().join(format!("File{}", ext.to_uppercase()));
        std::fs::write(&p, b"PK\x03\x04\xff").unwrap();
        let content = load(Source::Path(p), None).await.unwrap();
        assert_eq!(content.media_type, *media);
    }
}

#[test]
fn test_sniff_media_types() {
    let cases: &[(&[u8], Option<&str>)] = &[
        (b"%PDF-1.4", Some("application/pdf")),
        (b"\x89PNG....", Some("image/png")),
        (b"\xff\xd8\xff\xe0", Some("image/jpeg")),
        (b"GIF89a", Some("image/gif")),
        (b"ID3\x04", Some("audio/mpeg")),
        (b"fLaC", Some("audio/flac")),
        (b"OggS", Some("audio/ogg")),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", Some("image/webp")),
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", Some("audio/wav")),
        (b"RIFF\x00\x00\x00\x00AVI LIST", None),
        (b"\x00\x00\x00\x18ftypmp42", Some("video/mp4")),
        ("plain text, café".as_bytes(), Some("text/plain")),
        (b"\xff\xfe\x00\x01", None),
    ];
    for (bytes, expected) in cases {
        assert_eq!(sniff(bytes), *expected);
    }
}

#[tokio::test]
async fn test_bytes_uninferable_and_explicit_media_type_override() {
    let server = MockServer::start().await;
    mount_json(&server, 200, output("responses", None)).await;

    let err = extract::<Invoice>(
        configured(&server, "openai:test"),
        b"\xff\xfe\x00\x01".as_slice(),
        None,
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::InvalidConfig(_)));
    assert!(err.to_string().contains("media_type"));

    let _: Invoice = extract_with_options(
        configured(&server, "openai:test"),
        PDF_BYTES,
        ExtractOptions::new().with_media_type("text/plain"),
    )
    .await
    .unwrap();
    assert_eq!(
        last_document(&recorded_bodies(&server).await),
        &json!({"type": "input_text", "text": "%PDF-1.7 fake"})
    );

    let _: Invoice = extract_with_options(
        configured(&server, "openai:test"),
        "%PDF-1.7 fake",
        ExtractOptions::new().with_media_type("application/pdf"),
    )
    .await
    .unwrap();
    assert_eq!(
        last_document(&recorded_bodies(&server).await),
        &pdf_file_json()
    );
}

#[tokio::test]
async fn test_url_fetching_precedence_redirects_and_no_credentials() {
    let source_server = MockServer::start().await;
    let model_server = MockServer::start().await;
    mount_json(&model_server, 200, output("responses", None)).await;

    // HTML page fetched without Authorization header
    Mock::given(method("GET"))
        .and(path("/page"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/html; charset=utf-8")
                .set_body_bytes(b"<p>hi</p>"),
        )
        .mount(&source_server)
        .await;

    let url = format!("{}/page", source_server.uri());
    let _: Invoice = extract(configured(&model_server, "openai:test"), url, None)
        .await
        .unwrap();
    let source_reqs = source_server.received_requests().await.unwrap();
    assert!(source_reqs[0].headers.get("authorization").is_none());
    assert_eq!(
        last_document(&recorded_bodies(&model_server).await)["text"],
        "<p>hi</p>"
    );

    // Media type precedence: extension > header > generic/empty header -> sniff
    let prec_cases = [
        ("/invoice.pdf", Some("text/html"), "application/pdf"),
        (
            "/download-png",
            Some("image/png; charset=binary"),
            "image/png",
        ),
        (
            "/download-bin",
            Some("application/octet-stream"),
            "application/pdf",
        ),
        ("/download-none", None, "application/pdf"),
    ];
    for (route, ct, expected) in prec_cases {
        let mut resp = ResponseTemplate::new(200).set_body_bytes(PDF_BYTES);
        if let Some(h) = ct {
            resp = resp.insert_header("content-type", h);
        }
        Mock::given(method("GET"))
            .and(path(route))
            .respond_with(resp)
            .mount(&source_server)
            .await;
        let content = load(Source::Url(format!("{}{route}", source_server.uri())), None)
            .await
            .unwrap();
        assert_eq!(content.media_type, expected, "route {route}");
    }

    // Redirects are followed for URL sources
    Mock::given(method("GET"))
        .and(path("/start"))
        .respond_with(
            ResponseTemplate::new(302)
                .insert_header("location", format!("{}/final.txt", source_server.uri())),
        )
        .mount(&source_server)
        .await;
    Mock::given(method("GET"))
        .and(path("/final.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"Total due: 120.00"))
        .mount(&source_server)
        .await;
    let _: Invoice = extract(
        configured(&model_server, "openai:test"),
        format!("{}/start", source_server.uri()),
        None,
    )
    .await
    .unwrap();
    assert_eq!(
        last_document(&recorded_bodies(&model_server).await),
        &json!({"type": "input_text", "text": "Total due: 120.00"})
    );

    // HTTP 404 returns ExtractionError without echoing secret query params
    Mock::given(method("GET"))
        .and(path("/missing.pdf"))
        .respond_with(ResponseTemplate::new(404))
        .mount(&source_server)
        .await;
    let err = extract::<Invoice>(
        configured(&model_server, "openai:test"),
        format!("{}/missing.pdf?token=private-token", source_server.uri()),
        None,
    )
    .await
    .unwrap_err();
    assert!(matches!(err, Error::Extraction { .. }));
    let msg = err.to_string();
    assert!(msg.contains("Could not fetch the source URL (HTTP 404)"));
    assert!(msg.contains("Next step:"));
    assert!(!msg.contains("private-token"));
}

fn host_and_port(uri: &str) -> (String, u16) {
    let (host, port) = uri.trim_start_matches("http://").split_once(':').unwrap();
    (host.to_string(), port.parse().unwrap())
}

#[tokio::test]
async fn test_url_same_host_redirect_chain_is_capped() {
    let server = MockServer::start().await;
    let base = server.uri();

    // Three same-host hops are followed.
    for (from, to) in [("/s0", "/s1"), ("/s1", "/s2"), ("/s2", "/s3.txt")] {
        Mock::given(method("GET"))
            .and(path(from))
            .respond_with(
                ResponseTemplate::new(302).insert_header("location", format!("{base}{to}")),
            )
            .mount(&server)
            .await;
    }
    Mock::given(method("GET"))
        .and(path("/s3.txt"))
        .respond_with(ResponseTemplate::new(200).set_body_bytes(b"Total due: 120.00"))
        .mount(&server)
        .await;
    let content = load(Source::Url(format!("{base}/s0")), None).await.unwrap();
    assert_eq!(content.text(), "Total due: 120.00");

    // A fourth redirect is rejected and the final target is not requested.
    for (from, to) in [
        ("/f0", "/f1"),
        ("/f1", "/f2"),
        ("/f2", "/f3"),
        ("/f3", "/f4.txt"),
    ] {
        Mock::given(method("GET"))
            .and(path(from))
            .respond_with(
                ResponseTemplate::new(302).insert_header("location", format!("{base}{to}")),
            )
            .mount(&server)
            .await;
    }
    let err = load(Source::Url(format!("{base}/f0")), None)
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("too many redirects"), "{msg}");
    assert!(msg.contains("Next step:"));
    assert!(matches!(err, Error::Extraction { .. }));
    let requested: Vec<String> = server
        .received_requests()
        .await
        .unwrap()
        .iter()
        .map(|request| request.url.path().to_string())
        .collect();
    assert!(requested.contains(&"/f3".to_string()));
    assert!(!requested.iter().any(|path| path == "/f4.txt"));
}

#[tokio::test]
async fn test_url_redirects_reject_other_hosts_ports_credentials_and_loops() {
    let server = MockServer::start().await;
    let base = server.uri();
    let (host, port) = host_and_port(&base);

    let cases = [
        (
            "/cross",
            "http://example.invalid/private-redirect-token".to_string(),
            "original host",
        ),
        (
            "/creds",
            format!("http://user:super-secret@{host}:{port}/creds.txt"),
            "credentials",
        ),
        (
            "/port",
            format!("http://{host}:1/other-port"),
            "original port",
        ),
        ("/file", "file:///etc/passwd".to_string(), "http or https"),
        ("/loop", format!("{base}/loop"), "redirect loop"),
    ];
    for (route, location, needle) in cases {
        Mock::given(method("GET"))
            .and(path(route))
            .respond_with(ResponseTemplate::new(302).insert_header("location", location))
            .mount(&server)
            .await;
        let err = load(
            Source::Url(format!("{base}{route}?token=private-token")),
            None,
        )
        .await
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(needle), "{route}: {msg}");
        assert!(!msg.contains("private-token"), "{route}: {msg}");
        assert!(!msg.contains("super-secret"), "{route}: {msg}");
        assert!(!msg.contains("private-redirect-token"), "{route}: {msg}");
        assert!(!msg.contains("passwd"), "{route}: {msg}");
    }
}

#[tokio::test]
async fn test_url_content_length_over_the_download_limit_is_rejected_immediately() {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::time::Duration;

    use openextract::content::MAX_DOWNLOAD_BYTES;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let advertised = MAX_DOWNLOAD_BYTES + 1;
    std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        sock.set_read_timeout(Some(Duration::from_secs(5))).ok();
        let mut buf = [0u8; 2048];
        let _ = sock.read(&mut buf);
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n\
             Content-Length: {advertised}\r\nConnection: close\r\n\r\n"
        );
        let _ = sock.write_all(header.as_bytes());
        // Hold the connection. A client that buffers the body waits here.
        let _ = sock.read(&mut buf);
    });

    let result = tokio::time::timeout(
        Duration::from_secs(2),
        load(Source::Url(format!("http://{addr}/huge.bin")), None),
    )
    .await;
    let err = result
        .expect("Content-Length above the cap was not rejected before the body")
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("50 MiB download limit"), "{msg}");
    assert!(msg.contains("Next step:"));
    assert!(matches!(err, Error::Extraction { .. }));
}
