import com.datastax.spark.connector._
import play.api.libs.json.Json
import play.api.libs.json._
import java.io.{ByteArrayOutputStream, ByteArrayInputStream}
import java.util.zip.{GZIPOutputStream, GZIPInputStream}

case class RichheaderServiceName(service_name: String, sha256: String)
case class RichheaderSha256(sha256: String, service_name: String, results: Array[Byte])
case class RichheaderJoin(sha256: String, service_name: String, results: String)

val richheader_service_name_rdd = sc.cassandraTable[RichheaderServiceName](KEYSPACE, SERVICE_NAME_TABLE)
                                .where("service_name=?", "richheader")
                                .keyBy(x => (x.sha256, x.service_name));
val richheader_sha256_rdd = sc.cassandraTable[RichheaderSha256](KEYSPACE, SHA256_TABLE)
                          .where("service_name=?", "richheader")
                          .keyBy(x => (x.sha256, x.service_name));
val richheader_join = richheader_service_name_rdd.join(richheader_sha256_rdd)
                    .map(x => (new RichheaderJoin(x._1._1, x._1._2, unzip(x._2._2.results))))
                    .distinct();

richheader_join.toDF().write.format("parquet").save("richheader.df");
